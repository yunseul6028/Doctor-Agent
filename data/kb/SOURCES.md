# data/kb sources and attribution

Built by scripts/build_kb.py. See docs/licenses.md for the full license ledger.

- **DDXPlus**: DDXPlus Dataset (English) knowledge files: release_conditions.json, release_evidences.json. License: CC BY 4.0. https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585 — Fansi Tchango A, et al. DDXPlus: A New Dataset For Automatic Medical Diagnosis. NeurIPS 2022 Datasets and Benchmarks. Changes: evidences reduced to short terms and translated to Korean.
- **KCD**: 건강보험심사평가원_상병마스터 (KCD disease code master). License: KOGL Type 1 (공공누리 제1유형: 출처표시). https://www.data.go.kr/data/15067467/fileData.do — 출처: 건강보험심사평가원, 상병마스터 (공공데이터포털). 필요한 열만 발췌.
- **DO**: Human Disease Ontology (doid.obo). License: CC0 1.0. https://github.com/DiseaseOntology/HumanDiseaseOntology — Schriml LM, et al. Human Disease Ontology. disease-ontology.org
- **WD**: Wikidata structured data (P780 symptoms, P923 medical examinations, P699/P494/P486 ids, en/ko labels). License: CC0 1.0 (structured data namespaces). https://query.wikidata.org/ — Wikidata (CC0).
- **MedlinePlus**: MedlinePlus Health Topics XML (health topic summaries only). License: Public domain (U.S. government work; health topic summaries). A.D.A.M./ASHP content not used. https://medlineplus.gov/xml/mplus_topics_2026-09-26.xml — Courtesy of MedlinePlus from the National Library of Medicine. Symptoms/tests extracted and summarised in Korean by an LLM (see data/labels/kb_build_meta.json).
- **curated**: Our own Korean clinical wording tables: CURATED_SYN (build time) and src/doctor_agent/knowledge/kb_curated.py (runtime synonyms, phrase patterns, generic-term stop list, vital/lab thresholds, diagnosis-name spelling pairs). License: Self-authored.  — Written by the team; used only to match findings and diagnosis names to KB entries (adds no disease knowledge).
- **LLM**: Offline LLM processing (Korean labels, extraction from MedlinePlus). License: Our derived annotations.  — Generated offline; prompts, model and date in data/labels/kb_build_meta.json.

Versions: DO releases/2026-08-31/doid.obo; MedlinePlus XML 09/26/2026 02:30:41; built 2026-09-26.
Human Phenotype Ontology: not used.
