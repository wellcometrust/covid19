# Data

## Summary

These are some key points about the data to get you started.
For a more complete description you should read `metadata.readme`
and `json_schema.txt`.

The dataset consists of more than 30k publications that come mainly
from PMC (approx 27k) but also come from bioRxiv and medRxiv and CZI.
Each publication has a row on metadata.csv, the id that links the two
is the sha column that links to the name and paper_id of the pub. *Note*
however that additional material of a publication also has the same sha
as the publication. Another thing to point out is that the metadata
contain a url to the publication.

Publication keys
```
* abstract
* body_text
* back_matter
* metadata (authors, title)
* bib_entries (citations)
* ref_entries (tables, figures)
```

Abstract and body text
```
* text
* section (Introduction, Conclusion etc)
* cite_spans (start, end, text, ref_id)
* ref_spans
```

Metadata columns
```
1. cord_uid # unique identifier
2. sha
3. source_x
4. title
5. doi
6. pmcid
7. pubmed_id
8. license
9. abstract
10. publish_time
11. authors
12. journal
13. Microsoft Academic Paper ID
14. WHO #Covidence # identifier for CZI records
15. has_pdf_parse
16. has_pmc_xml_parse
17. full_text_file
18. url # url to the publication
```

## Questions

* Are there any duplicate publications?
* Are there publication that are not covid related?
* What is the meaning of the different subfolders (comm_use_subset, custom licence, etc)?

