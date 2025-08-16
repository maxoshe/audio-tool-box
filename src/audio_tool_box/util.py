def convert_db_to_factor(dbfs: float) -> float:
    return 10 ** (dbfs / 20)
