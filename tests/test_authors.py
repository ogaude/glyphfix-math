from lightonocr_math.authors import AUTHOR_RULE, author_keys, normalize_author, quarantine


def test_normalize_folds_and_splits_hyphens():
    assert normalize_author("José  Müller-Lüdenscheidt") == "jose muller ludenscheidt"


def test_normalize_drops_initial_only_and_single_token():
    assert normalize_author("J. Smith") is None
    assert normalize_author("Plato") is None
    assert normalize_author("J. R. Smith") is None


def test_normalize_keeps_middle_initial():
    assert normalize_author("John R. Smith") == "john r smith"


def test_author_keys_skips_ambiguous():
    assert author_keys(["Jane Doe", "J. Doe", "Plato"]) == {"jane doe"}


def test_rule_constants():
    assert AUTHOR_RULE["max_authors_per_paper"] == 50
    assert AUTHOR_RULE["min_tokens"] == 2


def _c(i, authors):
    return {"arxiv_id": i, "authors": authors}


def test_quarantine_full_name_only():
    cands = [
        _c("2301.00001", ["Jane Doe", "Bob Roe"]),
        _c("2301.00002", ["Jane Doe"]),
        _c("2301.00003", ["J. Doe"]),
        _c("2301.00004", ["Carl Other"]),
    ]
    q, stats = quarantine(cands, {"2301.00001"})
    assert q == {"2301.00002"}
    assert "2301.00001" not in q
    assert stats["quarantined"] == 1
    assert stats["top_names"][0]["name"] == "jane doe"


def test_quarantine_skips_large_author_papers():
    many = [f"Person{chr(97 + i % 26)}{i} Name{i}" for i in range(60)]
    cands = [
        _c("2301.00001", ["Jane Doe"]),
        _c("2301.00002", ["Jane Doe", *many]),
        _c("2301.00005", many),
        _c("2301.00006", many[:5]),
    ]
    # large candidate is never quarantined
    q, stats = quarantine(cands, {"2301.00001"})
    assert "2301.00002" not in q
    assert stats["skipped_large_author_papers"] == 2
    # large benchmark paper contributes no names
    q2, _ = quarantine(cands, {"2301.00005"})
    assert q2 == set()
