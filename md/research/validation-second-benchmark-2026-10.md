# A second ground-truth benchmark, and a real NISAR stack (#144, #141)

**Date:** 2026-10-31
**Method:** desk research with live, anonymous queries only: CMR, the ASF search API,
data.ca.gov CKAN, and NISAR docs at ASF. I didn't log in, use credentials, or download
anything that needs a login.

## #144: the mining-subsidence leveling benchmark

**The citation in the issue is wrong.** DOI 10.3390/land9080285 does not resolve in Crossref.
The paper the issue describes is Karamvasis & Karathanassi (2020), *Remote Sensing* 12(9):1380,
doi:10.3390/rs12091380. I read it via the authors' CC-BY copy (https://zenodo.org/record/6517008).

**Its ground truth isn't public.**
- The paper has no data-availability statement.
- The leveling is seven points in the Soulou area, surveyed June to December 2016.
- It was supplied by the Mines Central Support Department of PPC S.A.-Hellas and is cited as
  "personal communication, 2019".
- OPERA CSLC-S1 doesn't cover the site in the study period. Greece is outside the
  North-America production area. CMR holds only a one-month October 2023 sample there.

**A public substitute that fits better:** DWR California Aqueduct Subsidence Study, San Luis
Field Division, at https://data.ca.gov/dataset/california-aqueduct-subsidence-study (file
`caaq-slfd_dosamingos-kettlemen-ngvd29.xlsx`).
- **Benchmarks:** 448 on the canal lining and structures.
- **Survey epochs:** 2015 through 2025, eleven in all. 406 benchmarks have 8 or more of them.
- **Signal:** the largest 2015–2025 change is −0.70 m (36.109N, −120.066), the median −0.07 m.
- **Licence:** "No restrictions on public use".
- **OPERA coverage:** CMR returns 1,287 CSLC-S1 granules at the deepest benchmark, the first
  on 2016-07-05 (T042-088925-IW1).

The mechanism is aquifer compaction, not mining. The truth type is survey, independent of
InSAR, so it complements the GNSS harness.

Open point: the 2017 study report says the 2015 epoch was surveyed by GPS and earlier epochs by
precise leveling. The method for 2016–2025 is unconfirmed. Check the `Metadata` sheet before
treating it as leveling.

**Conclusion: the original dataset is not feasible. Adopt the aqueduct benchmark.** Fetching
the CSLC stack still needs an Earthdata login, so the build issue carries `needs-human-auth`.

## #141: a real multi-date NISAR stack

**Verdict: needs an Earthdata login. There is no anonymous path to the data.**
- **Search works anonymously.** CMR collection `NISAR_L2_GSLC_PROVISIONAL_V1`
  (`C2854332392-ASF`) holds 165,749 granules, 160,110 of them dated on or after 2026-06-17.
- **HTTPS download doesn't.** Anonymous HEAD on
  `https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GSLC_PROVISIONAL_V1/...` returns the
  size, but GET redirects to `urs.earthdata.nasa.gov` and ends in 401.
- **S3 doesn't either.** Anonymous `ListObjects` on the products bucket is denied. Temporary
  credentials need a login and only work in-region.
- **The docs agree.** https://nisar-docs.asf.alaska.edu/access-overview says download requires
  Earthdata Login.
- **A suitable stack exists.** Mexico City, track 113 D frame 079, has 9 dates (06-22 to 10-20)
  at about 22 GB per granule. It sits under the existing MMX1/ICMX/MXMX GNSS harness. I didn't
  check frame coverage of the exact station coordinates.
- **Read subsets, not whole files.** The stack is about 200 GB, so an AOI should be read with
  ranged reads (`/vsicurl/`) instead.
- **Results will need a re-run.** Provisional products are "calibrated and partially validated",
  and a validated reprocessing is planned for Q4 2026.

**Conclusion:** the stack is blocked on an Earthdata login. #141 stays open with
`needs-human-auth`.
