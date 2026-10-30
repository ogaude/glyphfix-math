# Error mining report

## Reading guide

- An error row is a training-split crop whose prediction does not render identically to the ground truth. 8,477 of 13,958 rows are error rows; train render-identical is 0.393, in line with the benchmark baseline (0.426).
- Candidate clusters only subdivide the `other` class. The glyph classes (prime_l, dropped_prime, greek_latin, decoration, i_1) keep their rows whatever you decide.
- Style rows (for example `\dfrac` -> `\frac`, `\mbox` -> `\text`) are still trained whether or not you accept a cluster. Accepting only gives a cluster its own equal share of the error rows.
- Spacing-only rows (same normalized LaTeX, different rendering) count as errors and are tagged; 1,104 rows carry the spacing-only diagnostic tag, and cluster c00 (NORM_EQUAL, 1,398 rows) is the broader layout candidate.
- The family column is a heuristic label. Clusters c04, c06, c14, c17, c18, c22 and c33 are `aligned` wrapper insertions or deletions, labelled glyph by the heuristic; they behave like layout.
- Rare-class supply (next table): prime_l, dropped_prime, greek_latin and i_1 fall short of the equal-share target of 533 rows. Under the approved default (distinct crops only, no repeats) the shortfalls are recorded, not padded.
- Fill the `decision` column or reply in the checkpoint with a list such as `left_right=c01,c02`.

Scored records: 13958; error rows (render differs): 8477; clean: 5481; ground truth uncompilable: 0; prediction errors: 0.
Spacing-only diagnostic tag: 1104 rows. Candidate families: {'layout': 1, 'style': 18, 'glyph': 13, 'other': 9}.

## Rare-class supply

| class | error rows | available after regression exclusion | equal-share target | short |
|---|---|---|---|---|
| prime_l | 10 | 3 | 533 | yes |
| dropped_prime | 151 | 122 | 533 | yes |
| greek_latin | 107 | 83 | 533 | yes |
| decoration | 1706 | 1676 | 533 | no |
| i_1 | 20 | 15 | 533 | yes |
| other | 6483 | 6483 | 533 | no |

## Candidate clusters (review)

| id | family | gt tokens | pred tokens | records | papers | available | examples | decision |
|---|---|---|---|---|---|---|---|---|
| c00 | layout | `NORM_EQUAL` | `NORM_EQUAL` | 1398 | 476 | 1398 | 008845d6cc5230fe, 00ab248340bd466c, 0107bd486644b38f, 0139219bfc31f1f5, 014cf9dc6cd2c2b5 |  |
| c01 | style | `` | `\right` | 1691 | 480 | 1691 | 000645985983d36e, 00082fe2f5263ba9, 000d9eabb4c3757f, 00148911f90f88b1, 004e8f86b197a332 |  |
| c02 | style | `` | `\left` | 1658 | 475 | 1658 | 000645985983d36e, 00082fe2f5263ba9, 000d9eabb4c3757f, 00148911f90f88b1, 004e8f86b197a332 |  |
| c03 | style | `` | `&` | 1108 | 408 | 1108 | 000645985983d36e, 007baffda9bd2167, 007f668c7ba8e9a1, 008074437a7cfb5e, 009ad747e41db053 |  |
| c04 | glyph | `` | `\begin { a l i g n e d }` | 666 | 312 | 666 | 009ad747e41db053, 00bae81080ea3a6b, 0102c56104aad693, 0113ec286892e3f4, 013d9861b58ef901 |  |
| c05 | style | `}` | `` | 534 | 236 | 534 | 00e202da73666897, 0102c56104aad693, 01b1bcdefa822e70, 01bab01cef0b1e3a, 026898b83fde27e3 |  |
| c06 | glyph | `` | `. \end { a l i g n e d }` | 407 | 234 | 407 | 009ad747e41db053, 00bae81080ea3a6b, 013d9861b58ef901, 01df9291a81f9863, 01f627fc3c551142 |  |
| c07 | style | `\right` | `` | 341 | 146 | 341 | 02a7e83a69e99229, 05cc4208286a77b5, 05f2ec965ed4279e, 05f62589ff1bc71a, 05f70e22fceb350c |  |
| c08 | style | `&` | `` | 333 | 160 | 333 | 005bcab0f880bbb2, 00ba0a75ddb17f74, 01b1bcdefa822e70, 022560b784886d1f, 02e4b43413de9f5b |  |
| c09 | style | `\left` | `` | 333 | 143 | 333 | 02a7e83a69e99229, 030b200ffdeb1b36, 05f62589ff1bc71a, 05f70e22fceb350c, 0659354bc9393772 |  |
| c10 | style | `{` | `` | 319 | 142 | 319 | 00e202da73666897, 01bab01cef0b1e3a, 026898b83fde27e3, 03255200a089214b, 032ce47e2e278837 |  |
| c11 | style | `` | `}` | 240 | 141 | 240 | 009ad747e41db053, 01df9291a81f9863, 020bfe4981200072, 0211a14ad5a58c11, 033d5f3b70d4cded |  |
| c12 | glyph | `s p` | `a` | 177 | 66 | 177 | 008074437a7cfb5e, 04665792a22b2f6a, 0ae8216b74dddceb, 0bab1521520b62f0, 0dd75e10284df167 |  |
| c13 | glyph | `t` | `g n e d` | 177 | 66 | 177 | 008074437a7cfb5e, 04665792a22b2f6a, 0ae8216b74dddceb, 0bab1521520b62f0, 0dd75e10284df167 |  |
| c14 | glyph | `` | `, \end { a l i g n e d }` | 170 | 111 | 170 | 000645985983d36e, 0102c56104aad693, 0201de251084de5f, 02b081db4bc509c6, 03090b55325f056d |  |
| c15 | style | `\dots` | `\ldots` | 133 | 87 | 133 | 007baffda9bd2167, 03ec22ef4a3b7c08, 08ae9da6a34af99a, 08cc97eceeaee1f4, 0b7a196818e283d3 |  |
| c16 | style | `\\` | `` | 132 | 84 | 132 | 0138fb4d9c0647d3, 0250bea6e9d71396, 05c4cba0ce9c556b, 05f62589ff1bc71a, 05f98ba907f96689 |  |
| c17 | glyph | `` | `\begin { a l i g n e d } &` | 127 | 86 | 127 | 000645985983d36e, 0211a14ad5a58c11, 0299b0d29343e012, 032ce47e2e278837, 0429061d48844917 |  |
| c18 | glyph | `` | `\end { a l i g n e d }` | 116 | 71 | 116 | 0113ec286892e3f4, 039e8de46e111d0c, 086b3dc7d9a7b7f8, 0b4f2b58f8a0e635, 0ccc4f7f2946d0b9 |  |
| c19 | style | `\dfrac` | `\frac` | 116 | 47 | 116 | 05db929846400b56, 05f62589ff1bc71a, 083d390e49070f63, 0ad5237441c19e12, 11fa9aaf6e3e8337 |  |
| c20 | style | `\limits` | `` | 113 | 37 | 113 | 0410685ac29fa23e, 041d24b224509e73, 086b3dc7d9a7b7f8, 0978ac16b4f84943, 0aae7e7f203ad7ba |  |
| c21 | other | `$` | `}` | 111 | 67 | 111 | 03588c73a1f27737, 03ec22ef4a3b7c08, 0411a941e0218b35, 0512a45f412cc673, 051eee3d8cc59ac4 |  |
| c22 | glyph | `` | `} . \end { a l i g n e d` | 110 | 78 | 110 | 025c8a01eab53113, 044a21000cbafa36, 049e4357c60a0f36, 05db929846400b56, 0cfd2be07893d0f3 |  |
| c23 | style | `` | `{` | 104 | 63 | 104 | 01df9291a81f9863, 0211a14ad5a58c11, 034f0d93616cce37, 0370b7c4bb76cc00, 04310f3d079a621c |  |
| c24 | other | `\coloneqq` | `: =` | 101 | 42 | 101 | 012db42e7f06f127, 02b8e40e66dffa3b, 086fc1003c9f6996, 08d4a0423acf9986, 0b2868eb04c531c6 |  |
| c25 | glyph | `\begin { s p l i t }` | `` | 90 | 49 | 90 | 01b1bcdefa822e70, 01bab01cef0b1e3a, 0b4068cff8a03634, 0bc1bcd7e9cceadf, 0c835a0153f5f01b |  |
| c26 | style | `\tfrac` | `\frac` | 86 | 41 | 86 | 044a21000cbafa36, 05ce73791ea198ec, 0bf4c95d892c1cdf, 0caee6cd39b495c1, 11ca63d26ad57db2 |  |
| c27 | other | `\| \|` | `\\|` | 86 | 20 | 86 | 0211a14ad5a58c11, 0370b7c4bb76cc00, 04c37c9078e942f6, 055dfea104da9192, 08cc670e6055ca90 |  |
| c28 | other | `\vert` | `\|` | 76 | 29 | 76 | 009ad747e41db053, 010304b196f7ea95, 028ddb8f1a635e5c, 033c23f3f2b7f7dd, 0d7e3ae8c2a3e679 |  |
| c29 | style | `\mbox` | `\text` | 75 | 37 | 75 | 01928a537ea10bd2, 0411a941e0218b35, 051eee3d8cc59ac4, 08e96d6bc8e202a2, 09ca5a319bf5545d |  |
| c30 | other | `\|` | `\mid` | 70 | 48 | 70 | 00c031edbf330175, 029637574ab803e2, 063da48a6dc0adc0, 09bfcd54ed4401f6, 0c4a084e93e0bc96 |  |
| c31 | other | `\left \{` | `` | 61 | 39 | 61 | 03588c73a1f27737, 03ec22ef4a3b7c08, 09c5007097aaaefa, 09e96161ba86ab53, 134184d232f894e1 |  |
| c32 | other | `\ast` | `*` | 61 | 28 | 61 | 00bae81080ea3a6b, 01928a537ea10bd2, 0731f4fa4dd2d69e, 0a9cbbee377a941d, 0c4a084e93e0bc96 |  |
| c33 | glyph | `\begin { a l i g n e d }` | `` | 60 | 35 | 60 | 046261aaed81c19f, 04be0d2c92c9d927, 0c312dd8bd4985a6, 0d46d272be0b91c0, 0e1928880e11ddf2 |  |
| c34 | glyph | `a y` | `i x` | 60 | 33 | 60 | 02a42764fb1313df, 0982a443dd80f28e, 0c1f30ddd440e7bd, 1226531a30b3cd94, 131a16d8ba373554 |  |
| c35 | glyph | `r` | `t` | 60 | 33 | 60 | 02a42764fb1313df, 0982a443dd80f28e, 0c1f30ddd440e7bd, 1226531a30b3cd94, 131a16d8ba373554 |  |
| c36 | other | `\star` | `*` | 60 | 16 | 60 | 04ca19f437a6417e, 0501b762d03a50e8, 0556657d14d04311, 1b1615720922674d, 208f8c0b3066db2f |  |
| c37 | glyph | `` | `c` | 59 | 36 | 59 | 03588c73a1f27737, 03ec22ef4a3b7c08, 09c5007097aaaefa, 09e96161ba86ab53, 134184d232f894e1 |  |
| c38 | style | `\dots` | `\cdots` | 58 | 33 | 58 | 005bcab0f880bbb2, 0ccc4f7f2946d0b9, 1627c9471fbdb1bb, 204a85f65013eb64, 225f6655c77d4408 |  |
| c39 | other | `$ }` | `` | 56 | 34 | 56 | 0411a941e0218b35, 0a1e2ced7af4814b, 1498b3edc98f3942, 1a6906f8be0e9be7, 1eecdfa24db5b357 |  |
| c40 | style | `` | `\\` | 55 | 42 | 55 | 025c8a01eab53113, 0a9cbbee377a941d, 0f091225ad82cd62, 0f79b9c243b78e2b, 10c2ce809613246b |  |
