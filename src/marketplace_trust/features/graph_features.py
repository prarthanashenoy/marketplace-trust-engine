"""
Graph-inspired feature engineering.

We do NOT use a Graph Neural Network here. Instead we extract
graph-structural signals as numeric features using pandas and NetworkX.

Why this approach:
- GNNs are harder to explain, harder to debug, and require specialised
  infrastructure. For a portfolio project this complexity adds noise.
- The relational signals we care about (shared devices, shared IPs,
  co-buyer networks) can be captured as simple aggregate counts and
  network centrality measures.

Features capture:
1. Shared device/IP clusters: how many seller accounts share the same device
   or IP hash as this seller's account?
2. Buyer-seller bipartite graph: are the same buyers transacting with multiple
   suspicious sellers? (co-buyer overlap)
3. Review network: are the same reviewers leaving reviews for the same sellers?

These features directly model the "fraud ring" and "fake review network"
scenarios from the data generator.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False
    logger.warning("networkx not installed; network centrality features will be skipped")


def build_graph_features(
    sellers: pd.DataFrame,
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    reviews: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-seller graph-inspired features.

    Parameters
    ----------
    sellers:      Sellers table (seller_id, user_id).
    users:        Users table (user_id, device_id).
    transactions: Transactions table (seller_id, buyer_id).
    reviews:      Reviews table (seller_id, buyer_id).
    events:       Events table (user_id, device_id, ip_hash).

    Returns
    -------
    DataFrame with one row per seller and graph-feature columns.
    """
    logger.info("Building graph features")

    feats = sellers[["seller_id", "user_id"]].copy()

    feats = feats.merge(_shared_device_features(sellers, users, events), on="seller_id", how="left")
    feats = feats.merge(_shared_ip_features(sellers, users, events), on="seller_id", how="left")
    feats = feats.merge(_buyer_overlap_features(transactions), on="seller_id", how="left")
    feats = feats.merge(_review_network_features(reviews), on="seller_id", how="left")

    if _HAS_NETWORKX:
        feats = feats.merge(_network_centrality_features(transactions), on="seller_id", how="left")

    numeric_cols = feats.select_dtypes(include="number").columns
    feats[numeric_cols] = feats[numeric_cols].fillna(0)

    # Drop helper column
    feats = feats.drop(columns=["user_id"], errors="ignore")

    logger.info("Graph features shape: %s", feats.shape)
    return feats


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _shared_device_features(
    sellers: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    How many seller accounts share the same device as this seller?

    A high count means a single physical device is being used to operate
    multiple seller accounts — a strong fraud ring signal.
    """
    # Get registration device per seller via their user_id
    seller_devices = sellers[["seller_id", "user_id"]].merge(
        users[["user_id", "device_id"]], on="user_id", how="left"
    )

    # Count how many sellers share each device_id
    device_seller_count = (
        seller_devices.groupby("device_id")["seller_id"]
        .count()
        .reset_index()
        .rename(columns={"seller_id": "sellers_sharing_device"})
    )
    result = seller_devices.merge(device_seller_count, on="device_id", how="left")

    # Also look at event-level device diversity: how many distinct devices
    # were used by the accounts that share a device with this seller?
    if len(events) > 0:
        # Devices seen in events associated with each seller's user
        event_devices = (
            events.groupby("user_id")["device_id"]
            .nunique()
            .reset_index()
            .rename(columns={"device_id": "distinct_event_devices"})
        )
        result = result.merge(event_devices, on="user_id", how="left")
        result["distinct_event_devices"] = result["distinct_event_devices"].fillna(0)
    else:
        result["distinct_event_devices"] = 0

    return result[["seller_id", "sellers_sharing_device", "distinct_event_devices"]]


def _shared_ip_features(
    sellers: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    How many distinct IP hashes are associated with this seller's account,
    and how many other sellers share those IPs?
    """
    if len(events) == 0:
        return sellers[["seller_id"]].assign(
            distinct_ips=0,
            max_accounts_per_ip=0,
        )

    seller_user = sellers[["seller_id", "user_id"]]

    # IPs per user in events
    user_ips = (
        events.groupby("user_id")["ip_hash"]
        .nunique()
        .reset_index()
        .rename(columns={"ip_hash": "distinct_ips"})
    )

    # For each IP, count distinct users
    ip_user_counts = (
        events.groupby("ip_hash")["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"user_id": "users_per_ip"})
    )

    # For each user, find the maximum accounts-per-ip across all IPs they used
    user_max_ip_count = (
        events.merge(ip_user_counts, on="ip_hash", how="left")
        .groupby("user_id")["users_per_ip"]
        .max()
        .reset_index()
        .rename(columns={"users_per_ip": "max_accounts_per_ip"})
    )

    result = seller_user.merge(user_ips, on="user_id", how="left").merge(
        user_max_ip_count, on="user_id", how="left"
    )
    return result[["seller_id", "distinct_ips", "max_accounts_per_ip"]]


def _buyer_overlap_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Measure buyer-seller overlap: do the same buyers transact with multiple
    sellers in this dataset?

    seller_buyer_overlap_count: number of buyers of this seller who also
    bought from at least one other seller. High overlap within a cluster
    is a fraud ring indicator.
    """
    if len(transactions) == 0:
        return pd.DataFrame(
            columns=["seller_id", "seller_buyer_overlap_count", "avg_buyer_seller_count"]
        )

    # For each buyer, count how many distinct sellers they've bought from
    buyer_seller_count = (
        transactions.groupby("buyer_id")["seller_id"]
        .nunique()
        .reset_index()
        .rename(columns={"seller_id": "num_sellers_bought_from"})
    )

    # Merge back to transactions and then aggregate per seller
    enriched = transactions.merge(buyer_seller_count, on="buyer_id", how="left")

    agg = (
        enriched.groupby("seller_id")
        .agg(
            seller_buyer_overlap_count=(
                "num_sellers_bought_from",
                lambda x: (x > 1).sum(),
            ),
            avg_buyer_seller_count=("num_sellers_bought_from", "mean"),
        )
        .reset_index()
    )
    return agg


def _review_network_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Detect fake review networks: same reviewers leaving reviews for the
    same set of sellers.

    shared_reviewer_count: number of reviewers who also reviewed at least
    one other seller. High values combined with high five_star_rate = fake.
    """
    if len(reviews) == 0:
        return pd.DataFrame(
            columns=["seller_id", "shared_reviewer_count", "avg_reviewer_seller_count"]
        )

    reviewer_seller_count = (
        reviews.groupby("buyer_id")["seller_id"]
        .nunique()
        .reset_index()
        .rename(columns={"seller_id": "num_sellers_reviewed"})
    )

    enriched = reviews.merge(reviewer_seller_count, on="buyer_id", how="left")

    agg = (
        enriched.groupby("seller_id")
        .agg(
            shared_reviewer_count=(
                "num_sellers_reviewed",
                lambda x: (x > 1).sum(),
            ),
            avg_reviewer_seller_count=("num_sellers_reviewed", "mean"),
        )
        .reset_index()
    )
    return agg


def _network_centrality_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute degree centrality in the buyer-seller bipartite transaction graph.

    Sellers with high degree centrality are connected to many buyers; sellers
    where the same buyers form a dense cluster (high clustering) are suspicious.

    Requires NetworkX.
    """
    assert _HAS_NETWORKX, "networkx required for centrality features"

    if len(transactions) == 0:
        return pd.DataFrame(columns=["seller_id", "seller_degree_centrality"])

    # Build bipartite graph: seller nodes prefixed 's_', buyer nodes prefixed 'b_'
    G = nx.Graph()
    for _, row in transactions[["seller_id", "buyer_id"]].drop_duplicates().iterrows():
        G.add_edge(f"s_{row['seller_id']}", f"b_{row['buyer_id']}")

    centrality = nx.degree_centrality(G)

    seller_ids = transactions["seller_id"].unique()
    rows = []
    for sid in seller_ids:
        node = f"s_{sid}"
        rows.append(
            {
                "seller_id": sid,
                "seller_degree_centrality": centrality.get(node, 0.0),
            }
        )

    return pd.DataFrame(rows)
