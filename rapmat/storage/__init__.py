from rapmat.storage.base import StructureDescriptor, StructureStore


def __getattr__(name: str):
    if name == "SOAPDescriptor":
        from rapmat.storage.descriptors import SOAPDescriptor

        return SOAPDescriptor
    if name == "SurrealDBStore":
        from rapmat.storage.surrealdb_store import SurrealDBStore

        return SurrealDBStore

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
