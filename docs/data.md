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

Note that the column sha may contain multiple documents separated by 
semicolon ;

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

## Inconsistencies

Paper id | Title in metadata | Title in publication
-------- | ----------------- | ---------------------
0b4b4e5bb8d0d5167eec1e203b5dad283bd364a5 | Doubling healthy lifespan using drug synergy | 2
60abca9911a64805e51aa8deb70bb238c2f3d414 | Influenza-Associated Mortality in Georgia (2009–2011) | )
05c7871a5da795da7c03748bb7eff551bc66b721 | " " | A
54521bd72c1242be4cc154f66ebaf5759478e4cf | How Should U.S. Hospitals Prepare for Coronavirus Disease 2019 (COVID-19)? | \
31ff32abd2da30cf9a0ed4414fb83e19eb874314 | Truth in the Details | \
97b02266fec0d45796c970161d3ebb4ac56b8f6f | Summary of Suggestions From the Task Force for Mass Critical Care Summit, January 26–27, 2007 | *

## Duplicates

There are 269 rows that mention the same sha. There is not duplicate pmcid.

## Covid publications

There are 36484 of 52097 that mention one of covid, corona, sars, mers.
The majority of the documents are pre 2020 as well. (How many?)

## Questions

* What is the meaning of the different subfolders (comm_use_subset, custom licence, etc)?
* What is the difference between pmc and pdf folders?

