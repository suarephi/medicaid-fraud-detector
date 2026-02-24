# Medicaid Fraud Detection Methodology

## Overview

This document describes the legal basis, data sources, threshold justifications, FCA materiality analysis, and investigation workflow for each of the six fraud signals detected by the Medicaid Provider Fraud Signal Detection Engine.

---

## Signal 1: Excluded Provider Still Billing

### Legal Basis

- **Primary FCA Theory**: 31 U.S.C. SS 3729(a)(1)(A) -- presenting false or fraudulent claims
- **Regulatory Prohibition**: 42 U.S.C. SS 1320a-7b(f) -- any claim submitted by a provider excluded under Section 1128 or 1128A of the Social Security Act is an automatic false claim
- **Exclusion Authority**: 42 U.S.C. SS 1320a-7 (mandatory exclusion) and SS 1320a-7a (permissive exclusion)
- **Case Law**: *United States ex rel. Aranda v. Community Psychiatric Centers* (9th Cir. 2019) -- billing by excluded providers constitutes per se FCA liability

### Data Sources

| Source | Regulatory Authority | Description |
|--------|---------------------|-------------|
| OIG LEIE Database | 42 U.S.C. SS 1320a-7 | List of Excluded Individuals/Entities maintained by OIG per statutory mandate |
| Medicaid Claims (SDUD/NADAC) | 42 CFR Part 431 Subpart Q | State-reported Medicaid utilization data under federal reporting requirements |

### Detection Logic

1. Match provider NPIs against the OIG LEIE exclusion list
2. Filter to providers with active exclusions (exclusion date set, no reinstatement date)
3. Identify any Medicaid claims submitted after the exclusion effective date
4. Aggregate post-exclusion payments for damages calculation

### Threshold Justification

No minimum threshold applies. Per OIG guidance (OIG Special Advisory Bulletin, May 2013), **any** claim submitted by or on behalf of an excluded individual constitutes a violation. The per se nature of this violation means every post-exclusion claim is automatically false regardless of whether services were actually rendered.

### FCA Materiality Analysis

**Severity: CRITICAL (per se violation)**

Under *Universal Health Services v. United States ex rel. Escobar*, 579 U.S. 176 (2016), the Supreme Court established that FCA liability requires materiality -- the false statement must be material to the government's payment decision. However, excluded provider billing is one of the clearest examples of per se materiality:

- CMS conditions of participation explicitly require providers not be excluded (42 CFR SS 455.104)
- The government has consistently refused to pay excluded providers
- Exclusion status is an express condition of payment, not merely a condition of participation
- OIG has issued Civil Monetary Penalty (CMP) actions in virtually every identified case

**Damages**: All payments made post-exclusion are recoverable. Under the FCA, damages are trebled (3x), plus statutory penalties of $13,946 to $27,894 per false claim (2024 adjustment per 28 CFR SS 85.5).

---

## Signal 2: Billing Volume Outlier

### Legal Basis

- **Primary FCA Theory**: 31 U.S.C. SS 3729(a)(1)(A) -- presenting false or fraudulent claims through medically unnecessary services
- **Regulatory Framework**: 42 U.S.C. SS 1320a-7a(a)(1)(A) -- civil monetary penalties for presenting claims for services not provided as claimed or that are medically unnecessary
- **Anti-Kickback Implications**: 42 U.S.C. SS 1320a-7b(b) -- if inflated volume results from prohibited referral arrangements
- **Medical Necessity Requirement**: 42 CFR SS 440.230(d) -- services must be sufficient in amount, duration, and scope to achieve their purpose

### Data Sources

| Source | Regulatory Authority | Description |
|--------|---------------------|-------------|
| Medicaid Claims Data | 42 CFR Part 431 Subpart Q | Provider-level payment and claim volume aggregations |
| NPPES Registry | 45 CFR Part 162.408 | Provider taxonomy codes and practice location for peer grouping |

### Detection Logic

1. Aggregate total Medicaid payments per provider NPI
2. Join with NPPES to obtain taxonomy code and state for peer grouping
3. Within each peer group (taxonomy + state), calculate the 99th percentile and median
4. Flag providers exceeding the 99th percentile threshold
5. Compute ratio-to-peer-median as a severity indicator

### Threshold Justification

- **Peer group minimum**: 5 providers required per group (prevents false positives from small specialties)
- **99th percentile**: Aligns with CMS Comparative Billing Reports (CBR) methodology, which uses percentile-based peer comparison as a standard program integrity tool
- **Severity threshold**: >5x peer median for "high" severity. CMS Program Integrity Manual (Chapter 4, SS 4.17) identifies providers billing at 3x+ peer averages as warranting focused review; we use 5x as a higher-confidence threshold

### FCA Materiality Analysis

**Severity: MEDIUM to HIGH (based on ratio-to-peer-median)**

Statistical outlier status alone does not establish FCA liability, but it is strong circumstantial evidence that supports further investigation:

- Billing patterns dramatically exceeding peers raise an inference of upcoding, unbundling, or services lacking medical necessity
- Under *Escobar*, claims for medically unnecessary services are material because CMS would not have paid had it known the services were unnecessary
- If volume is driven by kickback arrangements, the Anti-Kickback Statute (AKS) creates automatic FCA liability under 42 U.S.C. SS 1320a-7b(g) (AKS violation renders resulting claims false per se)
- **Estimated overpayment**: Amount billed above the 99th percentile threshold for the peer group

---

## Signal 3: Rapid Billing Escalation (Bust-Out Fraud)

### Legal Basis

- **Primary FCA Theory**: 31 U.S.C. SS 3729(a)(1)(A) -- presenting false claims through a bust-out scheme
- **Enrollment Integrity**: 42 CFR SS 424.530 -- CMS may deny enrollment to providers demonstrating patterns of abusive billing
- **Revocation Authority**: 42 CFR SS 424.535(a)(8) -- CMS may revoke billing privileges for abuse of billing privileges
- **Fraud Pattern Recognition**: OIG Work Plan identifies "newly enrolled providers with aberrant billing" as a priority target

### Data Sources

| Source | Regulatory Authority | Description |
|--------|---------------------|-------------|
| Medicaid Claims Data | 42 CFR Part 431 Subpart Q | Monthly payment progressions per provider |
| NPPES Registry | 45 CFR Part 162.408 | Provider Enumeration Date to identify newly registered providers |

### Detection Logic

1. Identify providers enumerated within 24 months of their first Medicaid claim
2. Compute month-over-month payment growth for the first 12 billing months
3. Calculate rolling 3-month average growth rate to smooth volatility
4. Flag providers where peak rolling 3-month average growth exceeds 200%

### Threshold Justification

- **24-month enrollment window**: Aligns with CMS heightened screening period for newly enrolled providers under 42 CFR SS 424.518 and the ACA SS 6401 enhanced enrollment screening requirements
- **200% rolling 3-month growth**: Based on OIG data analytics identifying bust-out schemes. Legitimate new practices typically show 20-50% monthly growth; sustained 200%+ growth over a 3-month rolling average is statistically anomalous and consistent with the rapid extraction phase of bust-out fraud
- **12-month observation window**: Bust-out schemes typically execute within 6-18 months; 12 months captures the acceleration phase

### FCA Materiality Analysis

**Severity: MEDIUM to HIGH (based on peak growth rate)**

Bust-out fraud is a recognized scheme type in OIG enforcement:

- The billing pattern itself (rapid escalation by new provider, then abandonment) is treated as circumstantial evidence of fraudulent intent
- Under *Escobar*, claims submitted as part of a scheme to defraud are material because they involve a deliberate course of conduct designed to extract payments the government would not have made
- CMS Program Integrity Manual Chapter 15 identifies rapid billing escalation as a key indicator for prepayment review
- **>500% growth**: Elevated to "high" severity as this exceeds any plausible legitimate practice growth pattern
- **Estimated overpayment**: Sum of payments during months where month-over-month growth exceeded 200%

---

## Signal 4: Workforce Impossibility

### Legal Basis

- **Primary FCA Theory**: 31 U.S.C. SS 3729(a)(1)(B) -- making or using false records or statements material to a false claim
- **False Statement Prohibition**: 42 U.S.C. SS 1320a-7b(a)(3) -- prohibition on knowingly making false statements or representations of material fact in connection with claims
- **Implied False Certification**: Under *Escobar*, claims impliedly certify compliance with conditions of payment, including that services were actually rendered

### Data Sources

| Source | Regulatory Authority | Description |
|--------|---------------------|-------------|
| Medicaid Claims Data | 42 CFR Part 431 Subpart Q | Monthly claim counts per organizational NPI |
| NPPES Registry | 45 CFR Part 162.408 | Entity Type Code to distinguish organizations from individuals |

### Detection Logic

1. Filter to organizational providers (Entity Type Code = 2)
2. Aggregate claims by NPI and month
3. Identify peak month claim volume per provider
4. Calculate implied claims per provider-hour: claims / (22 working days * 8 hours)
5. Flag providers where implied rate exceeds 6 claims per hour

### Threshold Justification

- **Entity Type 2 only**: Individual providers (Type 1) may legitimately supervise multiple service lines; organizational NPIs represent single billing entities
- **22 working days * 8 hours = 176 hours/month**: Standard business operating hours per BLS guidelines
- **6 claims per provider-hour**: Represents one claim every 10 minutes sustained over an entire month. CMS Medically Unlikely Edit (MUE) program and specialty-specific time-per-service benchmarks indicate that sustained rates above this level are physically impossible for virtually all service types. Even emergency departments average 2-3 patients/hour per provider.

### FCA Materiality Analysis

**Severity: HIGH (inherent impossibility)**

Physically impossible claim volumes constitute powerful FCA evidence:

- If a provider cannot have physically delivered the claimed services, the claims necessarily contain false records
- Under *Escobar*, materiality is established because the government would not pay for services that were never rendered
- The impossibility itself demonstrates the "knowing" scienter element -- claims at physically impossible volumes cannot be submitted in good faith
- Courts have recognized workforce impossibility as strong indicator of fabricated claims (*United States v. Patel*, *United States v. Paulus*)
- **Estimated overpayment**: (peak_claims - 1056) * (average payment per claim), where 1056 = 176 hours * 6 claims/hour

---

## Signal 5: Shared Authorized Official (Shell Entity Network)

### Legal Basis

- **Primary FCA Theory**: 31 U.S.C. SS 3729(a)(1)(C) -- conspiracy to submit false or fraudulent claims
- **Anti-Kickback Statute**: 42 U.S.C. SS 1320a-7b(b) -- patient steering and self-referrals among controlled entities may constitute prohibited remuneration
- **Criminal Conspiracy**: 18 U.S.C. SS 1347 (healthcare fraud) and 18 U.S.C. SS 371 (general conspiracy)
- **Utilization Review**: 42 CFR Part 456 -- per-provider utilization review requirements that shell structures may be designed to circumvent

### Data Sources

| Source | Regulatory Authority | Description |
|--------|---------------------|-------------|
| NPPES Registry | 45 CFR Part 162.408 | Authorized Official names linked to organizational NPIs |
| Medicaid Claims Data | 42 CFR Part 431 Subpart Q | Combined billing volume across controlled entities |

### Detection Logic

1. Extract Authorized Official names from NPPES for all organizational NPIs
2. Group NPIs by Authorized Official (Last Name + First Name)
3. Filter to officials controlling 5 or more NPIs
4. Sum total Medicaid payments across all controlled NPIs
5. Flag officials where combined billing exceeds $1,000,000

### Threshold Justification

- **5+ NPIs**: While some legitimate healthcare systems have multiple NPIs under one official, controlling 5+ separate billing entities is unusual and warrants scrutiny. OIG Provider Enrollment Compendium identifies "multiple entity control" as a screening criterion
- **$1M combined threshold**: Ensures materiality -- below this level, the economic incentive for shell entity fraud is limited and the pattern may reflect legitimate multi-location operations
- **>$5M for "high" severity**: Aligns with DOJ Civil Fraud Initiative guidelines that prioritize cases exceeding $5M in potential damages for civil FCA enforcement

### FCA Materiality Analysis

**Severity: MEDIUM to HIGH (based on combined billing)**

Shell entity networks present strong FCA indicators:

- Conspiracy liability under SS 3729(a)(1)(C) requires only an agreement to submit false claims plus one overt act
- Coordinated billing through shell entities is circumstantial evidence of both the agreement and knowing falsity
- If patient steering occurs between controlled entities, AKS violations create automatic FCA liability under 42 U.S.C. SS 1320a-7b(g)
- **Estimated overpayment**: Not calculated directly -- requires forensic accounting to determine what portion of billing is legitimate vs. fraudulent

---

## Signal 6: Geographic Implausibility (Phantom Home Health Services)

### Legal Basis

- **Primary FCA Theory**: 31 U.S.C. SS 3729(a)(1)(G) -- reverse false claims (concealing or improperly avoiding obligation to repay)
- **Stark Law**: 42 U.S.C. SS 1395nn(a) -- physician self-referral prohibition applicable to home health services
- **Home Health Requirements**: 42 CFR SS 440.70 -- home health services must be medically necessary and ordered by a physician
- **Electronic Visit Verification**: 21st Century Cures Act SS 12006 -- requires states to implement EVV for home health services

### Data Sources

| Source | Regulatory Authority | Description |
|--------|---------------------|-------------|
| Medicaid Claims Data | 42 CFR Part 431 Subpart Q | Home health claims with HCPCS codes, beneficiary counts, and claim counts |
| HCPCS Code Set | 42 CFR SS 414.40 | Home health service codes (G0151-G0162, G0299-G0300, S9122-S9124, T1019-T1022) |

### Detection Logic

1. Filter claims to home health HCPCS codes
2. Aggregate by provider NPI and month: total claims and unique beneficiaries
3. Compute beneficiary-to-claims ratio per provider-month
4. Flag provider-months where: claims > 100 AND ratio < 0.1

### Threshold Justification

- **Home health code filtering**: Uses CMS-defined HCPCS codes for home health services (G, S, and T codes) to target the specific service category most vulnerable to phantom billing
- **100 claims minimum**: Eliminates small providers where low ratios may be statistical noise
- **0.1 ratio threshold**: A ratio below 0.1 means fewer than 1 unique beneficiary per 10 claims. Home health visits are inherently patient-specific; legitimate providers should have ratios closer to 0.5-1.0. A ratio below 0.1 strongly suggests either repeated billing on the same few patients (phantom visits) or fabricated beneficiary encounters
- **<0.05 for "high" severity**: Extremely low ratios (<1 beneficiary per 20 claims) are virtually impossible in legitimate home health delivery

### FCA Materiality Analysis

**Severity: MEDIUM to HIGH (based on beneficiary-to-claims ratio)**

Phantom home health services are a persistent enforcement priority:

- OIG consistently identifies home health fraud as a top enforcement priority (OIG Work Plan, annual updates)
- Low beneficiary ratios combined with high claim volumes are a recognized indicator of ghost patient schemes
- Under *Escobar*, phantom services are inherently material -- the government would never knowingly pay for services not delivered
- Stark Law violations in home health referrals create automatic FCA liability when claims result from prohibited referrals
- 21st Century Cures Act EVV requirements provide an additional compliance framework whose violation strengthens FCA theories
- **Estimated overpayment**: Not calculated directly -- requires claim-level audit to determine which services were actually delivered

---

## Investigation Workflow Recommendations

### Triage Priority

1. **Immediate action**: Signal 1 (excluded providers) -- per se violations requiring no further evidence of fraud
2. **Priority investigation**: Signal 4 (workforce impossibility) and Signal 3 (bust-out fraud) -- strong prima facie indicators
3. **Targeted audit**: Signal 2 (billing outliers) and Signal 6 (geographic implausibility) -- require medical record review
4. **Complex investigation**: Signal 5 (shared officials) -- requires corporate structure analysis

### Referral Pathways

| Action | Entity | Reference |
|--------|--------|-----------|
| Report suspected fraud | OIG Hotline | 1-800-HHS-TIPS, oig.hhs.gov/fraud/report-fraud/ |
| State-level investigation | Medicaid Fraud Control Unit (MFCU) | Each state maintains an MFCU per 42 CFR Part 1007 |
| Provider enrollment review | Medicare Administrative Contractor (MAC) | Per 42 CFR SS 424.535 revocation authority |
| Claims audit | Recovery Audit Contractor (RAC) | Per Section 1893(h) of the Social Security Act |
| Data-driven investigation | Zone Program Integrity Contractor (ZPIC) / Unified Program Integrity Contractor (UPIC) | Per CMS Program Integrity Manual Chapter 4 |
| Prepayment review | State Medicaid Agency | Per 42 CFR SS 455.23 payment suspension authority |
| Civil FCA action | DOJ Civil Division | Per 31 U.S.C. SS 3730(a) government intervention authority |
| Qui tam filing | Private relator | Per 31 U.S.C. SS 3730(b) whistleblower provisions |

### Evidence Preservation

For all signals, the following documentation should be preserved for potential litigation:

1. Complete claims data extracts with provider and beneficiary identifiers
2. NPPES enrollment records at time of detection
3. LEIE database snapshots (Signal 1)
4. Peer group composition and statistical methodology documentation
5. Chain of custody records for all data sources
6. Analyst notes documenting detection rationale and investigative steps taken
