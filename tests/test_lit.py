import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "literature"))
import lit_search as L

PUBMED = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Digital twin <i>monitoring</i> test</ArticleTitle><Abstract><AbstractText>Abs one.</AbstractText></Abstract></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/abc</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2601.00001</id><title> A  change
 detection paper</title><published>2026-01-02T00:00:00Z</published><summary>Summary text</summary></entry></feed>"""


class T(unittest.TestCase):
    def test_pubmed(self):
        r = L.parse_pubmed_xml(PUBMED)
        self.assertEqual((r[0]["year"], r[0]["doi"], r[0]["id"]), ("2024", "10.1/abc", "PMID:123"))
        self.assertIn("monitoring", r[0]["title"])

    def test_arxiv(self):
        r = L.parse_arxiv_atom(ARXIV)
        self.assertEqual((r[0]["year"], r[0]["title"]), ("2026", "A change detection paper"))

    def test_json_parsers_and_dedupe(self):
        s2 = L.parse_semantic_scholar({"data": [{"title": "X", "year": 2023, "externalIds": {"DOI": "10.1/ABC"}, "paperId": "p"}]})
        ep = L.parse_europepmc({"resultList": {"result": [{"title": "X", "pubYear": "2023", "doi": "10.1/abc", "id": "1", "source": "PPR"}]}})
        self.assertEqual(len(L.dedupe(s2 + ep)), 1)


if __name__ == "__main__":
    unittest.main()
