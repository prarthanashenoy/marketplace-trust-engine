"""
Pydantic schemas for all marketplace entities.

These schemas define the expected shape of generated data and are used
for validation throughout the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    EMAIL = "email"
    PHONE = "phone"
    FULL = "full"


class SellerCategory(str, Enum):
    ELECTRONICS = "electronics"
    FASHION = "fashion"
    HOME = "home"
    SPORTS = "sports"
    COLLECTIBLES = "collectibles"
    BOOKS = "books"
    AUTOMOTIVE = "automotive"
    OTHER = "other"


class ListingStatus(str, Enum):
    ACTIVE = "active"
    SOLD = "sold"
    REMOVED = "removed"
    EXPIRED = "expired"


class TransactionStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    DISPUTED = "disputed"
    PENDING = "pending"


class PaymentMethod(str, Enum):
    CARD = "card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    WALLET = "wallet"


class DeviceType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class EventType(str, Enum):
    LOGIN = "login"
    LISTING_VIEW = "listing_view"
    SEARCH = "search"
    PURCHASE = "purchase"
    MESSAGE = "message"
    REVIEW = "review"
    PROFILE_UPDATE = "profile_update"
    LOGOUT = "logout"


class FraudScenario(str, Enum):
    NORMAL = "normal"
    NEW_ACCOUNT_ABUSE = "new_account_abuse"
    MASS_LISTING_ABUSE = "mass_listing_abuse"
    SCAM_LISTINGS = "scam_listings"
    PAYMENT_REFUND_ABUSE = "payment_refund_abuse"
    ACCOUNT_TAKEOVER = "account_takeover"
    FRAUD_RING = "fraud_ring"
    FAKE_REVIEW_NETWORK = "fake_review_network"
    DEVICE_SHARING = "device_sharing"
    MIXED_SUSPICIOUS = "mixed_suspicious"


class UserRecord(BaseModel):
    user_id: str
    account_created_at: datetime
    country: str
    device_id: str
    account_age_days: float
    verification_status: VerificationStatus
    is_seller: bool = False


class SellerRecord(BaseModel):
    seller_id: str
    user_id: str
    seller_created_at: datetime
    category: SellerCategory
    average_rating: float = Field(ge=1.0, le=5.0)
    number_of_listings: int = Field(ge=0)
    number_of_transactions: int = Field(ge=0)
    cancellation_rate: float = Field(ge=0.0, le=1.0)
    fraud_scenario: FraudScenario
    is_fraudulent: bool


class ListingRecord(BaseModel):
    listing_id: str
    seller_id: str
    category: SellerCategory
    price: float = Field(gt=0)
    market_price: float = Field(gt=0)
    created_at: datetime
    description_length: int = Field(ge=0)
    image_count: int = Field(ge=0)
    listing_status: ListingStatus


class TransactionRecord(BaseModel):
    transaction_id: str
    buyer_id: str
    seller_id: str
    listing_id: str
    amount: float = Field(gt=0)
    timestamp: datetime
    payment_method: PaymentMethod
    status: TransactionStatus
    cancellation_flag: bool
    refund_flag: bool


class ReviewRecord(BaseModel):
    review_id: str
    buyer_id: str
    seller_id: str
    transaction_id: str
    rating: float = Field(ge=1.0, le=5.0)
    timestamp: datetime
    text_length: int = Field(ge=0)


class DeviceRecord(BaseModel):
    device_id: str
    device_type: DeviceType
    country: str
    first_seen_at: datetime


class EventRecord(BaseModel):
    event_id: str
    user_id: str
    timestamp: datetime
    event_type: EventType
    ip_hash: str
    device_id: str
