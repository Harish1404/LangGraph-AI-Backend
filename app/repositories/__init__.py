"""
Data access for auth. One module per collection, Mongo-only, no business logic.

The split from app/services exists so the service layer can be read as policy
("is this person allowed to act?") without Mongo query syntax in the middle of
it, and so the query shapes live next to the indexes that make them fast.
"""
