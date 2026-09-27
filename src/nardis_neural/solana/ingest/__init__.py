"""Real-chain ingestion: decode Solana transactions into market events and stream them
into :class:`~nardis_neural.solana.brain.SolanaBrain` through a read-only RPC client."""

from nardis_neural.solana.ingest.decoder import TransactionDecoder, decode_transactions
from nardis_neural.solana.ingest.rpc import SolanaRpc
from nardis_neural.solana.ingest.stream import ChainStreamer, run_live

__all__ = ["ChainStreamer", "SolanaRpc", "TransactionDecoder", "decode_transactions", "run_live"]
