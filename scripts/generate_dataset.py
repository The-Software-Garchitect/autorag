#!/usr/bin/env python3
"""Generate the Acme Co handbook and its gold questions.

The script is the source of truth: every graded answer is a sentence (or a
linearized table row) that is written into a specific document. Re-run it
after editing a policy so the docs and eval stay aligned.

    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --out data/acme
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autorag.config import default_config  # noqa: E402
from autorag.models import Document  # noqa: E402
from autorag.pipeline.chunk import chunk_documents  # noqa: E402
from autorag.pipeline.generate import MIN_OVERLAP, rank_weight, select_unit_texts  # noqa: E402
from autorag.textutil import linearize_row, render_markdown_table, term_overlap, units_from_text  # noqa: E402

# fact: doc, heading, sentence, qtype, question, extra phrasings, stale (doc, heading, sentence) or None
Fact = tuple


def facts() -> list[dict]:
    rows: list[dict] = []

    def add(doc, heading, sentence, qtype, question, alts=(), stale=None):
        rows.append(
            {
                "doc": doc,
                "heading": heading,
                "sentence": sentence,
                "qtype": qtype,
                "question": question,
                "alts": list(alts),
                "stale": stale,
            }
        )

    # --- PTO 2025, with a 2023 twin that uses the same words and the old number ---
    add(
        "pto-policy-2025",
        "Annual allotment",
        "During the first year of service, employees receive 15 PTO days per year.",
        "conflict",
        "How many PTO days do employees receive during the first year of service?",
        ["During the first year of service, how many PTO days do employees receive?"],
        ("pto-policy-2023", "Annual allotment", "During the first year of service, employees receive 12 PTO days per year."),
    )
    add(
        "pto-policy-2025",
        "Annual allotment",
        "After 1 year of service, employees receive 18 PTO days per year.",
        "conflict",
        "After 1 year of service, how many PTO days do employees receive?",
        ["How many PTO days do employees receive after 1 year of service?"],
        ("pto-policy-2023", "Annual allotment", "After 1 year of service, employees receive 12 PTO days per year."),
    )
    add(
        "pto-policy-2025",
        "Annual allotment",
        "After 3 years of service, employees receive 22 PTO days per year.",
        "conflict",
        "How many PTO days after 3 years?",
        [
            "After 3 years of service, how many PTO days do employees receive?",
            "Once someone has 3 years of service, how many PTO days do employees receive?",
        ],
        ("pto-policy-2023", "Annual allotment", "After 3 years of service, employees receive 16 PTO days per year."),
    )
    add(
        "pto-policy-2025",
        "Annual allotment",
        "After 5 years of service, employees receive 25 PTO days per year.",
        "conflict",
        "After 5 years of service, how many PTO days do employees receive?",
        ["How many PTO days do employees receive after 5 years of service?"],
        ("pto-policy-2023", "Annual allotment", "After 5 years of service, employees receive 20 PTO days per year."),
    )
    add(
        "pto-policy-2025",
        "Carryover and accrual",
        "The maximum PTO carryover is 5 days.",
        "conflict",
        "What is the maximum PTO carryover?",
        ["How many days is the maximum PTO carryover?"],
        ("pto-policy-2023", "Carryover and accrual", "The maximum PTO carryover is 0 days."),
    )
    add(
        "pto-policy-2025",
        "Carryover and accrual",
        "PTO accrues monthly and is posted on the first of each month.",
        "factual",
        "Is PTO posted on the first of each month?",
        ["PTO accrues monthly and is posted on the first of each month, right?"],
    )
    add(
        "pto-policy-2025",
        "Requesting time off",
        "Request PTO in Workday at least 5 business days ahead when the request is for 2 or more days.",
        "factual",
        "How far ahead do I request PTO in Workday when the request is for 2 or more days?",
        ["When a PTO request is for 2 or more days, how many business days ahead do I request it in Workday?"],
    )
    add(
        "manager-time-off",
        "Approving requests",
        "Managers approve or decline PTO in Workday within 2 business days.",
        "factual",
        "How quickly should managers approve or decline PTO in Workday?",
        ["Within how many business days do managers approve or decline PTO in Workday?"],
    )

    add(
        "sick-leave",
        "Allotment",
        "Employees receive 10 sick days per year, separate from PTO.",
        "factual",
        "How many sick days do employees receive per year?",
        ["Are sick days separate from PTO, and how many sick days do employees receive per year?"],
    )
    add(
        "sick-leave",
        "Documentation",
        "A doctor's note is required after 3 consecutive sick days.",
        "factual",
        "When is a doctor's note required for sick days?",
        ["After how many consecutive sick days is a doctor's note required?"],
    )
    add(
        "sick-leave",
        "Allotment",
        "Unused sick days are not paid out and do not carry over.",
        "factual",
        "Do unused sick days carry over or get paid out?",
        ["Are unused sick days paid out?"],
    )

    add(
        "parental-leave",
        "Length of leave",
        "The birth parent receives 16 weeks of paid parental leave.",
        "factual",
        "How many weeks of paid parental leave does the birth parent receive?",
        ["What paid parental leave does the birth parent receive?"],
    )
    add(
        "parental-leave",
        "Length of leave",
        "The non-birth parent receives 8 weeks of paid parental leave.",
        "factual",
        "How many weeks of paid parental leave does the non-birth parent receive?",
        ["What paid parental leave does the non-birth parent receive?"],
    )
    add(
        "parental-leave",
        "Length of leave",
        "An adoptive parent receives 12 weeks of paid parental leave.",
        "factual",
        "How many weeks of paid parental leave does an adoptive parent receive?",
        ["What paid parental leave does an adoptive parent receive?"],
    )
    add(
        "parental-leave",
        "How to request",
        "Request parental leave in Workday under Time Off, using the Parental Leave type, at least 30 days before the expected start when the timing is foreseeable.",
        "factual",
        "How do I request parental leave?",
        [
            "Where do I request parental leave, and how many days before the expected start when the timing is foreseeable?",
            "How do I request parental leave in Workday?",
        ],
    )
    add(
        "parental-leave",
        "How to request",
        "Tell your manager and your HRBP about parental leave.",
        "factual",
        "Do I tell my manager and my HRBP about parental leave?",
        ["Should I tell my manager and my HRBP about parental leave?"],
    )

    add(
        "bereavement-and-jury",
        "Bereavement",
        "Bereavement leave is 5 paid days for an immediate family member.",
        "factual",
        "How many paid bereavement days are there for an immediate family member?",
        ["What bereavement leave is available for an immediate family member?"],
    )
    add(
        "bereavement-and-jury",
        "Bereavement",
        "Bereavement leave is 2 paid days for an extended family member.",
        "factual",
        "How many paid bereavement days are there for an extended family member?",
        ["What bereavement leave is available for an extended family member?"],
    )
    add(
        "bereavement-and-jury",
        "Jury duty",
        "Jury duty is paid for the full duration of the summons.",
        "factual",
        "Is jury duty paid for the full duration of the summons?",
        ["Does Acme pay jury duty for the full duration of the summons?"],
    )

    add(
        "holidays-2025",
        "Fixed holidays",
        "Juneteenth is a company holiday in 2025.",
        "factual",
        "Is Juneteenth a company holiday in 2025?",
        ["In 2025, is Juneteenth a company holiday?"],
    )
    add(
        "holidays-2025",
        "Fixed holidays",
        "The day after Thanksgiving is a company holiday in 2025.",
        "conflict",
        "Is the day after Thanksgiving a company holiday?",
        ["Is the day after Thanksgiving a company holiday in the current calendar?"],
        ("holidays-2024", "Fixed holidays", "The day after Thanksgiving is a company holiday in 2024 only on paper and was not observed."),
    )
    # The stale holiday sentence above is intentionally NOT a near-copy.
    # A true conflict twin is added next so retrieval has an old calendar to demote.
    add(
        "holidays-2025",
        "Floating holidays",
        "Employees receive 2 floating holidays each calendar year.",
        "factual",
        "How many floating holidays do employees receive each calendar year?",
        ["Employees receive how many floating holidays each calendar year?"],
    )
    add(
        "holidays-2025",
        "Floating holidays",
        "Floating holidays must be scheduled in Workday before November 15.",
        "factual",
        "When must floating holidays be scheduled in Workday?",
        ["By what date must floating holidays be scheduled in Workday?"],
    )
    add(
        "holidays-2025",
        "Fixed holidays",
        "Acme observes 11 fixed company holidays in 2025.",
        "conflict",
        "How many fixed company holidays does Acme observe?",
        ["How many fixed company holidays are on the current calendar?"],
        ("holidays-2024", "Fixed holidays", "Acme observes 10 fixed company holidays in the old calendar."),
    )

    add(
        "remote-work-2025",
        "On-site days",
        "Office-assigned employees work on site 3 days a week: Tuesday, Wednesday, and Thursday.",
        "conflict",
        "How many days a week do office-assigned employees work on site?",
        ["Which pattern do office-assigned employees work on site each week?"],
        (
            "remote-work-2023",
            "On-site days",
            "Office-assigned employees work on site 4 days a week: Monday, Tuesday, Wednesday, and Thursday.",
        ),
    )
    add(
        "remote-work-2025",
        "Fully remote",
        "Fully remote work is available after 6 months with manager approval.",
        "factual",
        "When is fully remote work available?",
        ["After how many months is fully remote work available with manager approval?"],
    )
    add(
        "remote-work-2025",
        "Core hours",
        "Core collaboration hours are 10am to 3pm in the employee's local time zone.",
        "factual",
        "What are the core collaboration hours?",
        ["When are core collaboration hours in the employee's local time zone?"],
    )
    add(
        "remote-work-2025",
        "Home office",
        "The one-time home office stipend is $500.",
        "conflict",
        "What is the one-time home office stipend?",
        ["How much is the one-time home office stipend?"],
        ("remote-work-2023", "Home office", "The one-time home office stipend is $250."),
    )

    add(
        "code-of-conduct",
        "Gifts",
        "The maximum value of a gift from a vendor is $50.",
        "factual",
        "What is the maximum value of a gift from a vendor?",
        ["How much can a gift from a vendor be worth?"],
    )
    add(
        "code-of-conduct",
        "Conflicts",
        "Employees must disclose a personal relationship with a vendor before any purchasing decision.",
        "factual",
        "When must employees disclose a personal relationship with a vendor?",
        ["Do employees disclose a personal relationship with a vendor before a purchasing decision?"],
    )
    add(
        "code-of-conduct",
        "Events",
        "A rideshare home from an Acme event with alcohol is reimbursed.",
        "factual",
        "Is a rideshare home from an Acme event with alcohol reimbursed?",
        ["Does Acme reimburse a rideshare home after an event with alcohol?"],
    )
    add(
        "harassment-and-ethics",
        "Reporting",
        "Report harassment to People or to ethics@acme.example.",
        "factual",
        "How do I report harassment?",
        ["Where do I report harassment, including the ethics email?"],
    )
    add(
        "harassment-and-ethics",
        "Reporting",
        "The ethics hotline is 1-800-555-0199.",
        "factual",
        "What is the ethics hotline?",
        ["The ethics hotline is what?"],
    )
    add(
        "harassment-and-ethics",
        "Retaliation",
        "Retaliation for a good-faith report is prohibited.",
        "factual",
        "Is retaliation for a good-faith report prohibited?",
        ["What is the rule on retaliation for a good-faith report?"],
    )

    add(
        "performance-reviews",
        "Cycle",
        "Performance reviews happen twice a year, in April and October.",
        "factual",
        "When do performance reviews happen?",
        ["How many times a year do performance reviews happen, and in which months?"],
    )
    add(
        "performance-reviews",
        "Ratings",
        "A performance rating of 3 is defined to mean the employee is meeting expectations.",
        "factual",
        "What does a performance rating of 3 mean?",
        ["A performance rating of 3 is defined to mean the employee is meeting expectations?"],
    )
    add(
        "performance-reviews",
        "Ratings",
        "The performance rating scale runs from 1 to 4.",
        "factual",
        "What is the performance rating scale?",
        ["The performance rating scale runs between which numbers?"],
    )

    add(
        "working-hours",
        "Schedule",
        "Standard full-time hours are 40 hours per week.",
        "factual",
        "How many hours per week are standard full-time hours?",
        ["What are the standard full-time hours per week?"],
    )
    add(
        "working-hours",
        "Overtime",
        "Non-exempt overtime is paid at 1.5 times the base rate after 40 hours in a week.",
        "factual",
        "How is non-exempt overtime paid after 40 hours in a week?",
        ["What is the overtime rate for non-exempt staff after 40 hours in a week?"],
    )
    add(
        "working-hours",
        "Overtime",
        "Overtime for non-exempt staff must be approved by the manager before it is worked.",
        "factual",
        "When must overtime for non-exempt staff be approved?",
        ["Who approves overtime for non-exempt staff, and when?"],
    )
    add(
        "working-hours",
        "Classification",
        "Software engineers at Acme are classified as exempt.",
        "factual",
        "Are software engineers classified as exempt?",
        ["Software engineers are classified as exempt?"],
    )

    add(
        "introductory-period",
        "Timing",
        "The introductory period is 90 days.",
        "factual",
        "The introductory period is how many days?",
        ["The introductory period is 90 days?"],
    )
    add(
        "introductory-period",
        "Benefits",
        "Benefits start on day 1, not after the introductory period.",
        "factual",
        "When do benefits start relative to the introductory period?",
        ["Do benefits start on day 1 or after the introductory period?"],
    )
    add(
        "internal-transfer",
        "Eligibility",
        "Employees may request an internal transfer after 12 months in the current role.",
        "factual",
        "When may employees request an internal transfer?",
        ["After how many months in the current role may employees request an internal transfer?"],
    )
    add(
        "internal-transfer",
        "Process",
        "Internal transfers require a conversation with the current manager and the People partner.",
        "factual",
        "Internal transfers require a conversation with whom?",
        ["Internal transfers require a conversation with the current manager and the People partner?"],
    )

    add(
        "dental-and-vision",
        "Dental",
        "Acme pays 100% of the employee dental premium.",
        "factual",
        "What share of the employee dental premium does Acme pay?",
        ["Does Acme pay the employee dental premium in full?"],
    )
    add(
        "dental-and-vision",
        "Dental",
        "Acme pays 50% of the dependent dental premium.",
        "factual",
        "What share of the dependent dental premium does Acme pay?",
        ["How much of the dependent dental premium does Acme pay?"],
    )
    add(
        "dental-and-vision",
        "Dental",
        "The dental annual maximum is $1,500 per person.",
        "factual",
        "What is the dental annual maximum per person?",
        ["How much is the dental annual maximum?"],
    )
    add(
        "dental-and-vision",
        "Vision",
        "The vision exam allowance is once every 12 months.",
        "factual",
        "How often is the vision exam allowance available?",
        ["The vision exam allowance is available once every how many months?"],
    )
    add(
        "dental-and-vision",
        "Vision",
        "The vision frame allowance is $150.",
        "factual",
        "What is the vision frame allowance?",
        ["How much is the vision frame allowance?"],
    )
    add(
        "dental-and-vision",
        "Vision",
        "The employee vision premium is $10 per month.",
        "factual",
        "What is the employee vision premium per month?",
        ["How much is the employee vision premium each month?"],
    )

    add(
        "retirement-401k",
        "Match",
        "Acme matches 100% of the first 4% of eligible pay.",
        "factual",
        "Acme matches what percent of the first 4% of eligible pay?",
        ["What is the 401k match on the first 4% of eligible pay?"],
    )
    add(
        "retirement-401k",
        "Match",
        "Acme matches 50% of the next 2% of eligible pay.",
        "factual",
        "Acme matches what percent of the next 2% of eligible pay?",
        ["What is the 401k match on the next 2% of eligible pay?"],
    )
    add(
        "retirement-401k",
        "Match",
        "A full company match requires the employee to contribute 6%, and Acme then contributes 5%.",
        "factual",
        "What employee contribution produces the full company 401k match, and what does Acme contribute?",
        ["To receive the full company match, what percent does the employee contribute and what percent does Acme contribute?"],
    )
    add(
        "retirement-401k",
        "Vesting",
        "401k vesting is 25% per year, and the employee is fully vested after 4 years.",
        "factual",
        "How does 401k vesting work, and when is the employee fully vested?",
        ["What is the 401k vesting schedule?"],
    )
    add(
        "retirement-401k",
        "Enrollment",
        "Employees may enroll in the 401k on day 1.",
        "factual",
        "When may employees enroll in the 401k?",
        ["Can employees enroll in the 401k on day 1?"],
    )

    add(
        "learning-stipend",
        "Amount",
        "The annual learning stipend is $1,500.",
        "factual",
        "What is the annual learning stipend?",
        ["The annual learning stipend is $1,500?"],
    )
    add(
        "learning-stipend",
        "Process",
        "Learning stipend requests above $200 are submitted in Workday before the purchase.",
        "factual",
        "When are learning stipend requests above $200 submitted?",
        ["Do learning stipend requests above $200 go in Workday before the purchase?"],
    )
    add(
        "wellness-stipends",
        "Wellness",
        "The monthly wellness stipend is $50.",
        "factual",
        "What is the monthly wellness stipend?",
        ["The monthly wellness stipend is $50?"],
    )
    add(
        "wellness-stipends",
        "Internet",
        "Remote employees receive a home internet stipend of $60 per month.",
        "factual",
        "What home internet stipend do remote employees receive?",
        ["How much is the home internet stipend for remote employees each month?"],
    )
    add(
        "wellness-stipends",
        "Phone",
        "On-call employees receive a phone stipend of $40 per month.",
        "factual",
        "What phone stipend do on-call employees receive?",
        ["How much is the phone stipend for on-call employees per month?"],
    )
    add(
        "employee-assistance",
        "Counseling",
        "The employee assistance program includes 6 free counseling sessions per issue.",
        "factual",
        "How many free counseling sessions does the employee assistance program include per issue?",
        ["What does the employee assistance program cover for counseling?"],
    )
    add(
        "employee-assistance",
        "Privacy",
        "EAP sessions are confidential and are not reported to the manager.",
        "factual",
        "Are EAP sessions reported to the manager?",
        ["Are EAP counseling sessions confidential?"],
    )
    add(
        "life-and-disability",
        "Life",
        "Acme provides company-paid life insurance equal to 1 times annual salary.",
        "factual",
        "How much company-paid life insurance does Acme provide?",
        ["Company-paid life insurance is equal to what multiple of annual salary?"],
    )
    add(
        "life-and-disability",
        "Life",
        "Employees may buy supplemental life insurance up to 4 times annual salary.",
        "factual",
        "How much supplemental life insurance may employees buy?",
        ["Supplemental life insurance can be purchased up to what multiple of annual salary?"],
    )
    add(
        "life-and-disability",
        "Disability",
        "Short-term disability pays 60% of base pay for up to 12 weeks.",
        "factual",
        "What does short-term disability pay, and for how long?",
        ["Short-term disability pays what percent of base pay, for up to how many weeks?"],
    )

    add(
        "hsa-and-fsa",
        "HSA",
        "The Basic health plan is the HSA-eligible high deductible plan.",
        "factual",
        "Which health plan is the HSA-eligible high deductible plan?",
        ["Is the Basic health plan the HSA-eligible high deductible plan?"],
    )
    add(
        "hsa-and-fsa",
        "HSA",
        "Acme contributes $500 to an individual HSA and $1,000 to a family HSA each year.",
        "factual",
        "How much does Acme contribute to an individual HSA and a family HSA each year?",
        ["What are the annual Acme contributions to an individual HSA and a family HSA?"],
    )
    add(
        "hsa-and-fsa",
        "FSA",
        "The healthcare FSA annual limit at Acme is $3,300.",
        "factual",
        "What is the healthcare FSA annual limit?",
        ["How much is the healthcare FSA annual limit at Acme?"],
    )
    add(
        "hsa-and-fsa",
        "FSA",
        "Plus and Premium health plans can use the healthcare FSA, and the Basic health plan cannot.",
        "factual",
        "Which health plans can use the healthcare FSA?",
        ["Can the Basic health plan use the healthcare FSA?"],
    )

    add(
        "benefits-enrollment",
        "New hires",
        "New hires enroll in benefits within 30 days after the start date.",
        "factual",
        "How long after the start date do new hires have to enroll in benefits?",
        ["New hires enroll in benefits within how many days after the start date?"],
    )
    add(
        "benefits-enrollment",
        "New hires",
        "Missing the new-hire benefits window means waiting for open enrollment unless there is a qualifying life event.",
        "factual",
        "Missing the new-hire benefits window means waiting for what?",
        ["Missing the new-hire benefits window means waiting for open enrollment unless there is a qualifying life event?"],
    )
    add(
        "open-enrollment",
        "Dates",
        "Open enrollment for 2026 coverage runs from November 1 through November 15, 2025.",
        "factual",
        "When is open enrollment for 2026 coverage?",
        ["Open enrollment for 2026 coverage runs between which dates?"],
    )

    add(
        "expense-limits",
        "Meals",
        "The client dinner expense limit is $75 per person.",
        "factual",
        "What's the client dinner expense limit?",
        ["What is the client dinner expense limit per person?", "How much is the per-person client dinner expense limit?"],
    )
    add(
        "expense-limits",
        "Meals",
        "The team meal expense limit is $35 per person.",
        "factual",
        "What is the team meal expense limit per person?",
        ["How much is the team meal expense limit?"],
    )
    add(
        "expense-limits",
        "Travel meals",
        "When per diem is not used, the travel breakfast limit is $20, the travel lunch limit is $30, and the travel dinner limit is $50.",
        "factual",
        "When per diem is not used, what are the travel breakfast, lunch, and dinner limits?",
        ["What is the travel dinner limit when per diem is not used?"],
    )
    add(
        "expense-limits",
        "Receipts",
        "Receipts are required for expenses over $25.",
        "factual",
        "When are receipts required for expenses?",
        ["Receipts are required for expenses over what amount?"],
    )
    add(
        "expense-limits",
        "Alcohol",
        "Alcohol is reimbursable only alongside a customer meal and only inside that cap.",
        "factual",
        "Is alcohol reimbursable only alongside a customer meal?",
        ["Alcohol is reimbursable only alongside a customer meal and only inside that cap?"],
    )

    add(
        "expense-approval",
        "Thresholds",
        "Expenses of $100 or less do not need pre-approval.",
        "factual",
        "Do expenses of $100 or less need pre-approval?",
        ["Which expenses do not need pre-approval?"],
    )
    add(
        "expense-approval",
        "Thresholds",
        "Expenses from $100.01 through $1,000 require manager approval.",
        "factual",
        "Which expenses require manager approval?",
        ["Expenses from $100.01 through $1,000 require whose approval?"],
    )
    add(
        "expense-approval",
        "Thresholds",
        "Directors and Finance both approve an expense above $1,000.",
        "factual",
        "Who must approve an expense above $1,000?",
        ["Directors and Finance both approve an expense above $1,000?"],
    )

    add(
        "travel-policy",
        "Flights",
        "Book flights at least 14 days before departure when the trip is known that far ahead.",
        "factual",
        "How far ahead should flights be booked when the trip is already known?",
        ["When a trip is known early, flights are booked at least how many days before departure?"],
    )
    add(
        "travel-policy",
        "Cabin",
        "Premium economy requires director approval on flights over 6 hours.",
        "factual",
        "When does premium economy require director approval?",
        ["Premium economy on flights over 6 hours requires whose approval?"],
    )
    add(
        "travel-policy",
        "Hotels",
        "Book hotels through the travel desk so the stay stays inside the city lodging per diem.",
        "factual",
        "How should hotels be booked relative to the city lodging per diem?",
        ["Where do employees book hotels so the stay stays inside the lodging per diem?"],
    )

    add(
        "corporate-card",
        "Statements",
        "Corporate card statements must be coded in the expense tool by the 5th business day of the next month.",
        "factual",
        "When must corporate card statements be coded in the expense tool?",
        ["Corporate card statements are coded by which business day of the next month?"],
    )
    add(
        "corporate-card",
        "Personal charges",
        "Personal charges on the corporate card must be repaid within 10 business days.",
        "factual",
        "How quickly must personal charges on the corporate card be repaid?",
        ["Personal charges on the corporate card are repaid within how many business days?"],
    )
    add(
        "mileage-and-ground",
        "Mileage",
        "The mileage reimbursement rate is $0.70 per mile.",
        "factual",
        "What is the mileage reimbursement rate?",
        ["How much is mileage reimbursement per mile?"],
    )
    add(
        "mileage-and-ground",
        "Rideshare",
        "Rideshare trips require a receipt when the fare is over $25.",
        "factual",
        "When do rideshare trips require a receipt?",
        ["A rideshare receipt is required when the fare is over what amount?"],
    )

    add(
        "laptop-standard",
        "Standard build",
        "The standard laptop is a 14 inch MacBook Pro or a 14 inch Dell XPS with 32 GB of memory and a 512 GB drive.",
        "factual",
        "Is the standard laptop a MacBook Pro or a Dell XPS?",
        ["The standard laptop is a 14 inch MacBook Pro or a 14 inch Dell XPS?"],
    )
    add(
        "laptop-standard",
        "Upgrades",
        "Laptop upgrades beyond the standard build require IT approval and a manager note.",
        "factual",
        "What is required for a laptop upgrade beyond the standard build?",
        ["Who has to approve a laptop upgrade beyond the standard build?"],
    )
    add(
        "laptop-standard",
        "Refresh",
        "Laptops are replaced on a 3 year cycle.",
        "factual",
        "How often are laptops replaced?",
        ["Laptops are replaced on what cycle?"],
    )
    add(
        "vpn-guide",
        "Client",
        "The VPN client is Cisco AnyConnect.",
        "factual",
        "Which VPN client does Acme use?",
        ["What is the name of the VPN client?"],
    )
    add(
        "vpn-guide",
        "Gateway",
        "The VPN gateway hostname is vpn.acme.example.",
        "factual",
        "What is the VPN gateway hostname?",
        ["Which hostname do I enter for the VPN gateway?"],
    )
    add(
        "vpn-guide",
        "Access",
        "VPN access requires multifactor authentication.",
        "factual",
        "Does VPN access require multifactor authentication?",
        ["VPN access requires multifactor authentication?"],
    )
    add(
        "passwords-and-mfa",
        "Length",
        "Passwords must be at least 14 characters.",
        "factual",
        "Passwords must be at least how many characters?",
        ["Passwords must be at least 14 characters?"],
    )
    add(
        "passwords-and-mfa",
        "Manager",
        "Acme uses 1Password as the company password manager.",
        "factual",
        "What is the company password manager?",
        ["Which password manager does Acme use?"],
    )
    add(
        "passwords-and-mfa",
        "Rotation",
        "Passwords are rotated every 180 days.",
        "factual",
        "How often are passwords rotated?",
        ["Passwords are rotated every how many days?"],
    )
    add(
        "passwords-and-mfa",
        "MFA",
        "Multifactor authentication is required for every company application.",
        "factual",
        "Is multifactor authentication required for every company application?",
        ["Which company applications require multifactor authentication?"],
    )
    add(
        "passwords-and-mfa",
        "Reuse",
        "Employees may not reuse any of their last 8 passwords.",
        "factual",
        "May employees reuse any of their last 8 passwords?",
        ["Employees may not reuse any of their last 8 passwords?"],
    )

    add(
        "first-week-checklist",
        "Day 1",
        "On day 1, pick up your badge and laptop from the office front desk or from the shipped kit.",
        "factual",
        "Where do I pick up my badge and laptop on day 1?",
        ["On day 1, what do I pick up from the front desk or the shipped kit?"],
    )
    add(
        "first-week-checklist",
        "Day 1",
        "Complete 1Password and Slack setup on day 1.",
        "factual",
        "Do I complete 1Password and Slack setup on day 1?",
        ["Complete 1Password and Slack setup on day 1?"],
    )
    add(
        "first-week-checklist",
        "Manager",
        "Schedule a manager one-on-one within the first 3 business days.",
        "factual",
        "When do I schedule a manager one-on-one?",
        ["A manager one-on-one is scheduled within how many business days?"],
    )
    add(
        "security-training",
        "Deadline",
        "New hires finish security training within 5 business days after the start date.",
        "factual",
        "When do new hires finish security training?",
        ["New hires finish security training within how many business days after the start date?"],
    )
    add(
        "security-training",
        "Phishing",
        "Phishing drills are run by Security about once a quarter, and reporting a drill is never punished.",
        "factual",
        "How often does Security run phishing drills, and is reporting a drill punished?",
        ["Are employees punished for reporting a phishing drill?"],
    )
    add(
        "software-catalog",
        "Standard tools",
        "The standard software catalog includes Google Workspace, Slack, Zoom, GitHub, Figma, 1Password, and Workday.",
        "factual",
        "What is in the standard software catalog?",
        ["Which tools are included in the standard software catalog?"],
    )
    add(
        "software-catalog",
        "Extra software",
        "Extra software with a fee above $200 needs manager approval in the IT portal.",
        "factual",
        "When does extra software need manager approval?",
        ["Extra software above what fee needs manager approval in the IT portal?"],
    )
    add(
        "it-support",
        "Contact",
        "IT support email is helpdesk@acme.example.",
        "factual",
        "What is the IT support email?",
        ["Which email reaches IT support?"],
    )
    add(
        "it-support",
        "Slack",
        "Urgent IT help is handled in the Slack channel it-help.",
        "factual",
        "Where is urgent IT help handled in Slack?",
        ["Which Slack channel handles urgent IT help?"],
    )
    add(
        "it-support",
        "Hours",
        "IT support hours are 8am to 6pm Central Time on business days.",
        "factual",
        "What are IT support hours?",
        ["When is IT support available on business days?"],
    )

    add(
        "about-acme",
        "Company",
        "Acme Co was founded in 2014.",
        "factual",
        "When was Acme founded?",
        ["In what year was Acme Co founded?"],
    )
    add(
        "about-acme",
        "Product",
        "Acme builds Acme Flow, a workflow product for mid-market operations teams.",
        "factual",
        "What is Acme Flow?",
        ["Acme builds which workflow product?"],
    )
    add(
        "about-acme",
        "Size",
        "Acme has about 420 employees.",
        "factual",
        "Acme has about 420 employees?",
        ["Does the company have about 420 employees?"],
    )
    add(
        "teams",
        "List",
        "The teams at Acme are Platform, Product, Design, People, Finance, Customer Success, Security, and Marketing.",
        "factual",
        "The teams at Acme include Platform, Product, and Design?",
        ["The teams at Acme are Platform, Product, Design, People, Finance, Customer Success, Security, and Marketing?"],
    )
    add(
        "teams",
        "Customer Success",
        "Customer Success sits in Austin and Chicago.",
        "factual",
        "Customer Success sits in Austin and Chicago?",
        ["Customer Success sits where?"],
    )
    add(
        "offices",
        "Austin",
        "The Austin headquarters address is 600 Congress Avenue, Suite 1800, Austin, TX 78701.",
        "factual",
        "What is the Austin headquarters address?",
        ["The Austin headquarters address is where?"],
    )
    add(
        "offices",
        "Chicago",
        "The Chicago office address is 233 South Wacker Drive, Suite 5400, Chicago, IL 60606.",
        "factual",
        "What is the Chicago office address?",
        ["Where is the Chicago office?"],
    )
    add(
        "offices",
        "Denver",
        "The Denver office address is 1801 California Street, Suite 900, Denver, CO 80202.",
        "factual",
        "What is the Denver office address?",
        ["Where is the Denver office?"],
    )
    add(
        "offices",
        "Reception",
        "Reception hours are 8:30am to 5:00pm local time.",
        "factual",
        "What are reception hours?",
        ["Reception hours are 8:30am to 5:00pm local time?"],
    )
    add(
        "key-contacts",
        "Leadership",
        "The CEO is Priya Raman, and her email is priya.raman@acme.example.",
        "factual",
        "Who is the CEO, and what is her email?",
        ["What is Priya Raman's role and email?"],
    )
    add(
        "key-contacts",
        "Leadership",
        "The Head of People is Luis Ortega, and his email is luis.ortega@acme.example.",
        "factual",
        "Who is the Head of People, and what is his email?",
        ["How do I email Luis Ortega?"],
    )
    add(
        "key-contacts",
        "Leadership",
        "The CFO is Helen Cho, and her email is helen.cho@acme.example.",
        "factual",
        "Who is the CFO, and what is her email?",
        ["How do I email Helen Cho?"],
    )
    add(
        "key-contacts",
        "Inboxes",
        "Payroll email is payroll@acme.example.",
        "factual",
        "What is the payroll email?",
        ["The payroll email is what?"],
    )
    add(
        "payroll",
        "Schedule",
        "The pay schedule is biweekly on Fridays.",
        "factual",
        "What is the pay schedule?",
        ["The pay schedule is biweekly on Fridays?"],
    )
    add(
        "payroll",
        "Direct deposit",
        "Set up direct deposit in Workday by the Wednesday before the first payday.",
        "factual",
        "When do I set up direct deposit?",
        ["Direct deposit is set up in Workday by which day before the first payday?"],
    )
    add(
        "visitors",
        "Badges",
        "Visitors sign in at reception and wear a paper badge for the entire visit.",
        "factual",
        "Visitors sign in at reception and wear what?",
        ["Visitors sign in at reception and wear a paper badge for the entire visit?"],
    )
    add(
        "visitors",
        "Escort",
        "A host employee must escort a visitor outside the reception area.",
        "factual",
        "Must a host employee escort a visitor outside the reception area?",
        ["A host employee must escort a visitor outside the reception area?"],
    )
    add(
        "glossary",
        "Terms",
        "PTO means paid time off.",
        "factual",
        "PTO means what?",
        ["PTO means paid time off?"],
    )
    add(
        "glossary",
        "Terms",
        "HRBP means the Human Resources business partner assigned to a team.",
        "factual",
        "HRBP means what?",
        ["HRBP means the Human Resources business partner assigned to a team?"],
    )
    add(
        "glossary",
        "Terms",
        "M&IE means meals and incidental expenses.",
        "factual",
        "M&IE means what?",
        ["M&IE means meals and incidental expenses?"],
    )
    add(
        "glossary",
        "Terms",
        "HDHP means high deductible health plan.",
        "factual",
        "HDHP means what?",
        ["HDHP means high deductible health plan?"],
    )
    add(
        "manager-time-off",
        "First week",
        "Managers hold a first-week one-on-one within 3 business days of a new hire's start.",
        "factual",
        "When does a manager hold a first-week one-on-one?",
        ["Managers hold a new hire one-on-one within how many business days of the start?"],
    )
    return rows


def tables() -> list[dict]:
    return [
        {
            "doc": "health-plans",
            "heading": "Monthly premiums and deductibles",
            "headers": [
                "Plan",
                "Employee monthly cost",
                "Family monthly cost",
                "Individual deductible",
                "HSA eligible",
            ],
            "rows": [
                ["Basic health plan", "$80", "$320", "$2,000", "Yes"],
                ["Plus health plan", "$160", "$540", "$1,000", "No"],
                ["Premium health plan", "$250", "$780", "$500", "No"],
            ],
            "questions": [
                (0, "What is the employee monthly cost for the Basic health plan?", []),
                (1, "What is the family monthly cost for the Plus health plan?", []),
                (2, "What is the employee monthly cost for the Premium health plan?", []),
                (2, "Which health plan has an individual deductible of $500?", ["The individual deductible of $500 belongs to which health plan?"]),
                (0, "What is the individual deductible for the Basic health plan?", []),
            ],
        },
        {
            "doc": "per-diem-rates",
            "heading": "City rates",
            "headers": [
                "City",
                "Lodging per diem",
                "Meals and incidental expenses per diem",
                "Notes",
            ],
            "rows": [
                [
                    "Austin",
                    "$180",
                    "$64",
                    "The Austin cap assumes a downtown hotel within walking distance of the headquarters.",
                ],
                [
                    "Denver",
                    "$190",
                    "$69",
                    "Denver lodging should be booked near Union Station when the meeting is downtown.",
                ],
                [
                    "New York",
                    "$280",
                    "$79",
                    "New York lodging fills early, so the travel desk holds a backup hotel.",
                ],
                [
                    "San Francisco",
                    "$270",
                    "$79",
                    "San Francisco lodging near the ferry building is usually at the cap.",
                ],
                [
                    "London",
                    "$310",
                    "$90",
                    "London lodging is booked in zone 1 only when the client site is there.",
                ],
                [
                    "Chicago",
                    "$220",
                    "$74",
                    "Chicago lodging per diem covers a downtown hotel near the river.",
                ],
            ],
            "questions": [
                (0, "What is the lodging per diem in Austin?", []),
                (1, "What is the lodging per diem in Denver?", []),
                (2, "What is the lodging per diem in New York?", []),
                (3, "What is the lodging per diem in San Francisco?", []),
                (4, "What is the lodging per diem in London?", []),
                (5, "What is the lodging per diem in Chicago?", ["How much is the Chicago lodging per diem?"]),
                (5, "What is the meals and incidental expenses per diem in Chicago?", []),
            ],
        },
    ]


DOC_META = {
    "pto-policy-2025": ("hr/pto-policy-2025.md", "Paid Time Off Policy", "current", "2025-01-01", "People", "pto-policy-2023"),
    "pto-policy-2023": ("hr/pto-policy-2023.md", "Paid Time Off Policy (2023)", "superseded", "2023-01-01", "People", None),
    "sick-leave": ("hr/sick-leave.md", "Sick Leave", "current", "2025-01-01", "People", None),
    "parental-leave": ("hr/parental-leave.md", "Parental Leave", "current", "2025-01-01", "People", None),
    "bereavement-and-jury": ("hr/bereavement-and-jury.md", "Bereavement and Jury Duty", "current", "2025-01-01", "People", None),
    "holidays-2025": ("hr/holidays-2025.md", "Company Holidays 2025", "current", "2025-01-01", "People", "holidays-2024"),
    "holidays-2024": ("hr/holidays-2024.md", "Company Holidays 2024", "superseded", "2024-01-01", "People", None),
    "remote-work-2025": ("hr/remote-work-2025.md", "Remote and Hybrid Work", "current", "2025-01-01", "People", "remote-work-2023"),
    "remote-work-2023": ("hr/remote-work-2023.md", "Remote and Hybrid Work (2023)", "superseded", "2023-03-01", "People", None),
    "code-of-conduct": ("hr/code-of-conduct.md", "Code of Conduct", "current", "2024-06-01", "People", None),
    "harassment-and-ethics": ("hr/harassment-and-ethics.md", "Harassment Reporting and Ethics", "current", "2024-06-01", "People", None),
    "performance-reviews": ("hr/performance-reviews.md", "Performance Reviews", "current", "2025-01-01", "People", None),
    "working-hours": ("hr/working-hours.md", "Working Hours and Overtime", "current", "2025-01-01", "People", None),
    "introductory-period": ("hr/introductory-period.md", "Introductory Period", "current", "2025-01-01", "People", None),
    "internal-transfer": ("hr/internal-transfer.md", "Internal Transfers", "current", "2025-01-01", "People", None),
    "manager-time-off": ("hr/manager-time-off.md", "Manager Guide to Time Off", "current", "2025-01-01", "People", None),
    "health-plans": ("benefits/health-plans.md", "Medical Plans", "current", "2025-01-01", "People", None),
    "dental-and-vision": ("benefits/dental-and-vision.md", "Dental and Vision", "current", "2025-01-01", "People", None),
    "retirement-401k": ("benefits/retirement-401k.md", "401k Retirement Plan", "current", "2025-01-01", "People", None),
    "learning-stipend": ("benefits/learning-stipend.md", "Learning Stipend", "current", "2025-01-01", "People", None),
    "wellness-stipends": ("benefits/wellness-stipends.md", "Wellness, Internet, and Phone Stipends", "current", "2025-01-01", "People", None),
    "employee-assistance": ("benefits/employee-assistance.md", "Employee Assistance Program", "current", "2024-01-01", "People", None),
    "life-and-disability": ("benefits/life-and-disability.md", "Life Insurance and Disability", "current", "2025-01-01", "People", None),
    "hsa-and-fsa": ("benefits/hsa-and-fsa.md", "HSA and FSA", "current", "2025-01-01", "People", None),
    "benefits-enrollment": ("benefits/benefits-enrollment.md", "Benefits Enrollment", "current", "2025-01-01", "People", None),
    "open-enrollment": ("benefits/open-enrollment.md", "Open Enrollment", "current", "2025-09-01", "People", None),
    "expense-limits": ("expenses/expense-limits.md", "Expense Limits", "current", "2025-01-01", "Finance", None),
    "expense-approval": ("expenses/expense-approval.md", "Expense Approval", "current", "2025-01-01", "Finance", None),
    "travel-policy": ("expenses/travel-policy.md", "Travel Policy", "current", "2025-01-01", "Finance", None),
    "per-diem-rates": ("expenses/per-diem-rates.md", "Per Diem Rates", "current", "2025-01-01", "Finance", None),
    "corporate-card": ("expenses/corporate-card.md", "Corporate Card", "current", "2025-01-01", "Finance", None),
    "mileage-and-ground": ("expenses/mileage-and-ground.md", "Mileage and Ground Transport", "current", "2025-01-01", "Finance", None),
    "laptop-standard": ("it/laptop-standard.md", "Laptop Standard", "current", "2025-02-01", "IT", None),
    "vpn-guide": ("it/vpn-guide.md", "VPN Access", "current", "2025-02-01", "IT", None),
    "passwords-and-mfa": ("it/passwords-and-mfa.md", "Passwords and Multifactor Authentication", "current", "2025-02-01", "Security", None),
    "first-week-checklist": ("it/first-week-checklist.md", "First Week Checklist", "current", "2025-01-01", "IT", None),
    "software-catalog": ("it/software-catalog.md", "Software Catalog", "current", "2025-01-01", "IT", None),
    "it-support": ("it/it-support.md", "IT Support", "current", "2025-01-01", "IT", None),
    "security-training": ("it/security-training.md", "Security Training", "current", "2025-01-01", "Security", None),
    "about-acme": ("org/about-acme.md", "About Acme", "current", "2025-01-01", "Company", None),
    "teams": ("org/teams.md", "Teams", "current", "2025-01-01", "Company", None),
    "offices": ("org/offices.md", "Offices", "current", "2025-01-01", "Company", None),
    "key-contacts": ("org/key-contacts.md", "Key Contacts", "current", "2025-01-01", "Company", None),
    "payroll": ("org/payroll.md", "Payroll", "current", "2025-01-01", "Finance", None),
    "visitors": ("org/visitors.md", "Visitors and Badges", "current", "2024-05-01", "Workplace", None),
    "glossary": ("org/glossary.md", "Handbook Glossary", "current", "2025-01-01", "People", None),
}

INTRO = {
    "pto-policy-2025": "Paid time off is here so people actually rest. Tenure is measured from the full-time start date, and the current allotment below replaces every earlier intranet page. Managers should plan coverage before they approve a long stretch.",
    "pto-policy-2023": "Superseded on December 31, 2024. This page is kept so old links do not 404. Do not use it for a current balance. The questions people still paste into Slack are answered below with the 2023 numbers.",
    "sick-leave": "Sick time covers illness and medical appointments. It is not a substitute for PTO and it is not a performance mark when it is used honestly.",
    "parental-leave": "Parental leave is paid and job-protected. Coordinate the dates with your manager early so the team can cover the work, and keep People in the thread.",
    "bereavement-and-jury": "These leaves are separate from PTO and sick time. Tell your manager as soon as you can; you do not need to share more detail than you want to.",
    "holidays-2025": "The office is closed on the fixed holidays below. Floating holidays belong to the employee, not the team calendar.",
    "holidays-2024": "Superseded calendar. Kept because a few offer letters from that year still link here. The current holiday list is the 2025 policy.",
    "remote-work-2025": "Most roles are hybrid. The pattern below is the default for anyone assigned to an office. Fully remote is an exception with an owner, not a vibe.",
    "remote-work-2023": "Superseded remote policy. The on-site expectation and the stipend changed when the 2025 policy took effect. This copy remains because an old onboarding deck still links to it.",
    "code-of-conduct": "We work with customers, vendors, and each other. The short version: be honest, skip the gray-area gift, and don't put a friend on a contract.",
    "harassment-and-ethics": "You can report to your manager, to People, or straight to the ethics line. You do not have to have a perfect writeup to start.",
    "performance-reviews": "Reviews are a written record of how the work went, not a surprise. Your manager owes you a draft before the calibration meeting.",
    "working-hours": "Full-time work is a week, not a hero streak. Non-exempt teammates must be paid for extra hours, and exempt engineers are still expected to live inside a normal week.",
    "introductory-period": "The first months are for learning the codebase and the customers. They are not a benefits waiting room.",
    "internal-transfer": "Moving teams is normal. Doing it in the first months of a role usually is not, because the team that hired you is still ramping you.",
    "manager-time-off": "Approvals sit with the manager. People Ops watches the queue only when a request has been waiting.",
    "health-plans": "Three medical plans are on offer. Premiums below are the employee cost per month after Acme's share. Compare deductibles before you pick on premium alone.",
    "dental-and-vision": "Dental and vision sit beside the medical plan. You can elect them even if you waive medical.",
    "retirement-401k": "The 401k is through the company plan. Match is based on eligible pay, and you can change the deferral in the plan site.",
    "learning-stipend": "Books, courses, and conference tickets that make you better at the job belong here. Team offsites do not.",
    "wellness-stipends": "Wellness, home internet, and on-call phones are separate stipends. They do not come out of the learning budget.",
    "employee-assistance": "The EAP is a counseling benefit run by an outside vendor. Acme pays the invoice and does not see the notes.",
    "life-and-disability": "Basic life and short-term disability are automatic. Buy-up life is optional and billed through payroll.",
    "hsa-and-fsa": "An HSA and a healthcare FSA are not the same account, and they do not pair with every medical plan. Read the plan name before you elect one.",
    "benefits-enrollment": "New hires get a window at the start. After that, changes wait for open enrollment or a qualifying life event such as a birth or a marriage.",
    "open-enrollment": "Open enrollment is the annual chance to change medical, dental, vision, and the FSA. Elections lock in when the window closes.",
    "expense-limits": "Spend company money as if finance will read the receipt out loud. Limits are per person, not per table, unless the line says otherwise.",
    "expense-approval": "The approval chain depends on the amount. Pre-approval and the later expense report are different steps.",
    "travel-policy": "Book the boring option. The travel desk exists so a last-minute fare does not become your problem or finance's.",
    "per-diem-rates": "Use these city rates instead of the standard travel meal caps when you are staying overnight. Lodging is a nightly cap before tax.",
    "corporate-card": "The corporate card is for company spend. It is not a backup personal card that you sort out later.",
    "mileage-and-ground": "Drive your own car only when it is the sensible way to the airport or a customer site. Log miles from the office or from home, whichever is shorter.",
    "laptop-standard": "Everyone gets a standard build. Special hardware is for roles that compile or design all day, and it is requested, not assumed.",
    "vpn-guide": "The VPN is required off the office network. Do not invent a gateway hostname from an old screenshot.",
    "passwords-and-mfa": "Passwords live in the company manager. Reuse and short passwords are the two ways accounts actually get popped.",
    "first-week-checklist": "The first week is a checklist, not a vibe. Finish the items that unblock payroll, chat, and the badge before you start a project.",
    "software-catalog": "These tools are already licensed. Ask before you put a new vendor on a card.",
    "it-support": "IT can reset a laptop, a badge, and an account. They cannot change a PTO balance.",
    "security-training": "Security training is short and mandatory. The phishing drills are practice, and reporting one is the outcome Security wants.",
    "about-acme": "Acme is a mid-size software company. The product helps operations teams move work across tools without a pile of custom scripts.",
    "teams": "Chapters and guilds are not separate headcount, so they are not listed here.",
    "offices": "Three offices. Austin is the headquarters. Badges from one office work at the others.",
    "key-contacts": "Use these inboxes before you guess a person's direct email from a conference slide. Leadership emails below are the public internal addresses.",
    "payroll": "Pay is biweekly. If a deposit looks wrong, write payroll with the pay period and a screenshot before you post in a large channel.",
    "visitors": "Customers and candidates sign in. A visitor without a host does not get past reception.",
    "glossary": "Short definitions for the acronyms the rest of the handbook uses without explaining.",
    "slack-ask-hr": "Unofficial export from the #ask-hr Slack channel. People paste old answers here. The linked policy pages win when they disagree.",
    "slack-ask-it": "Unofficial export from #ask-it. Hostnames in old threads go stale. Use the VPN guide.",
}


def expand_question(question: str, alts: list[str]) -> list[str]:
    lowered = question[0].lower() + question[1:]
    # Prefixes are stopwords only, so they don't change the overlap score.
    forms = [question, f"According to the handbook, {lowered}", f"Hey, {lowered}"]
    forms.extend(alts)
    seen = set()
    unique = []
    for form in forms:
        key = form.strip()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def build_sections(fact_rows: list[dict]) -> dict[str, dict[str, list[str]]]:
    sections: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for fact in fact_rows:
        bucket = sections[fact["doc"]][fact["heading"]]
        if fact["sentence"] not in bucket:
            bucket.append(fact["sentence"])
        if fact["stale"]:
            doc, heading, sentence = fact["stale"]
            stale_bucket = sections[doc][heading]
            if sentence not in stale_bucket:
                stale_bucket.append(sentence)
    return sections


def render_body(doc_id: str, sections: dict, table_by_doc: dict) -> str:
    parts = [INTRO[doc_id]]
    for heading, sentences in sections.get(doc_id, {}).items():
        parts.append(f"## {heading}")
        parts.extend(sentences)
    if doc_id in table_by_doc:
        table = table_by_doc[doc_id]
        parts.append(f"## {table['heading']}")
        parts.append(render_markdown_table(table["headers"], table["rows"]))
        parts.append(
            "Read the row that matches the plan or the city. Do not average across rows, and do not use a superseded rate sheet."
        )
    if doc_id == "pto-policy-2023":
        parts.append("## Questions people still ask")
        parts.append("Q: How many PTO days after 3 years? A: 16 PTO days.")
        parts.append("Q: What is the maximum PTO carryover? A: 0 days.")
        parts.append("Q: How many days a week did office-assigned employees work on site under this era? See the 2023 remote page, not this one.")
    if doc_id == "holidays-2025":
        parts.append("## The closed days")
        parts.append(
            "Closed days in the current year include New Year's Day, Martin Luther King Jr Day, Presidents Day, Memorial Day, Independence Day, Labor Day, Thanksgiving Day, Christmas Eve, and Christmas Day."
        )
    return "\n\n".join(parts).strip() + "\n"


def frontmatter(doc_id: str) -> dict:
    _path, title, status, effective, department, supersedes = DOC_META[doc_id]
    meta = {
        "id": doc_id,
        "title": title,
        "status": status,
        "effective_date": effective,
        "department": department,
        "version": effective[:4],
    }
    if supersedes:
        meta["supersedes"] = supersedes
    return meta


def dump_markdown(meta: dict, body: str) -> str:
    yaml_text = yaml.safe_dump(meta, sort_keys=False).strip()
    return f"---\n{yaml_text}\n---\n\n{body.strip()}\n"


def slack_hr() -> str:
    body = """Unofficial export. Timestamps are local to the poster. Do not cite this channel as policy.

**devon** [2023-06-01 11:02]
how many PTO days after 3 years? I swear HR said 16 PTO days. After 3 years of service, employees receive 16 PTO days per year. the maximum PTO carryover is 0 days.

**jordan** [2024-11-02 09:14]
that thread is from 2023 and people keep quoting it. I still see both numbers floating around.

**samira** [2024-11-02 09:22]
stop pasting the old doc. the current PTO page is the one dated 2025. this channel is not the policy.

**devon** [2023-08-14 16:40]
also the one-time home office stipend is $250. at least it was when I bought the desk.

**priya.patel** [2025-01-06 10:05]
reminder that #ask-hr is peer answers. if a number matters, open the handbook page with status current.
"""
    meta = {
        "id": "slack-ask-hr",
        "title": "#ask-hr Slack export",
        "status": "unofficial",
        "effective_date": "2024-11-02",
        "department": "People",
        "version": "export",
    }
    return dump_markdown(meta, INTRO["slack-ask-hr"] + "\n\n" + body)


def slack_it() -> str:
    body = """Unofficial export from #ask-it. Several of these hostnames are retired.

**mina** [2023-04-19 08:12]
vpn still the old gateway? I have been typing vpn.acme.internal into AnyConnect and it sometimes connects.

**chris** [2023-04-19 08:18]
yeah the VPN gateway hostname is vpn.acme.internal. don't use the example one, that was a placeholder in the draft.

**leah** [2025-02-03 13:44]
that reply is stale. the gateway changed. read the VPN guide before you paste a hostname into a ticket.

**mina** [2023-09-01 17:02]
also if your password is 8 characters you're fine, the 14 character thing was a proposal.

**leah** [2025-02-03 13:51]
also stale. password rules are whatever the current password page says, not this thread.
"""
    meta = {
        "id": "slack-ask-it",
        "title": "#ask-it Slack export",
        "status": "unofficial",
        "effective_date": "2025-02-03",
        "department": "IT",
        "version": "export",
    }
    return dump_markdown(meta, INTRO["slack-ask-it"] + "\n\n" + body)


def duplicate_of(source_id: str, new_id: str, title: str, path: str, body: str) -> tuple[str, str]:
    meta = {
        "id": new_id,
        "title": title,
        "status": "duplicate",
        "effective_date": "2025-01-01",
        "department": "People",
        "version": "copy",
        "canonical_id": source_id,
    }
    banner = (
        f"Duplicate copy of {source_id}. The canonical page is the one with status current and this id. "
        "This file exists because the old wiki mirrored policies into a second folder."
    )
    return path, dump_markdown(meta, banner + "\n\n" + body)


def build_questions(fact_rows: list[dict], table_rows: list[dict]) -> list[dict]:
    questions = []
    counter = 1

    def push(question, answer, sources, qtype, unanswerable=False):
        nonlocal counter
        questions.append(
            {
                "id": f"q{counter:04d}",
                "question": question,
                "answer": answer,
                "source_docs": sources,
                "type": qtype,
                "unanswerable": unanswerable,
            }
        )
        counter += 1

    for fact in fact_rows:
        for question in expand_question(fact["question"], fact["alts"]):
            push(question, fact["sentence"], [fact["doc"]], fact["qtype"])

    for table in table_rows:
        for index, question, alts in table["questions"]:
            answer = linearize_row(table["headers"], table["rows"][index])
            for form in expand_question(question, alts):
                push(form, answer, [table["doc"]], "table")

    # Multi-document questions. Both sentences are also standalone facts above.
    multi = [
        (
            "After the start date, within how many days do new hires finish security training and enroll in benefits?",
            [
                "New hires finish security training within 5 business days after the start date.",
                "New hires enroll in benefits within 30 days after the start date.",
            ],
            ["security-training", "benefits-enrollment"],
            ["Within how many days after the start date do new hires finish security training and enroll in benefits?"],
        ),
        (
            "What is the client dinner expense limit above the team meal?",
            [
                "The client dinner expense limit is $75 per person.",
                "The team meal expense limit is $35 per person.",
            ],
            ["expense-limits", "expense-limits"],
            [],
        ),
        (
            "What are the annual learning stipend and the monthly wellness stipend?",
            [
                "The annual learning stipend is $1,500.",
                "The monthly wellness stipend is $50.",
            ],
            ["learning-stipend", "wellness-stipends"],
            [],
        ),
        (
            "PTO days after 3, and the maximum PTO carryover?",
            [
                "After 3 years of service, employees receive 22 PTO days per year.",
                "The maximum PTO carryover is 5 days.",
            ],
            ["pto-policy-2025", "pto-policy-2025"],
            [],
        ),
    ]
    for question, sentences, sources, alts in multi:
        answer = " ".join(sentences)
        # de-dupe sources while keeping order
        ordered = list(dict.fromkeys(sources))
        for form in expand_question(question, alts):
            push(form, answer, ordered, "multi")

    unanswerable = [
        "Does the handbook include pet insurance for employees?",
        "What is the sabbatical policy?",
        "What is the CEO's home address?",
        "How many stock options do new hires receive?",
        "Does Acme offer a company car?",
        "What is the office wifi password in Chicago?",
        "Is there a commuter benefit for a Tokyo office?",
        "What is the CEO's personal phone number?",
        "Do we match charitable donations?",
        "What is the dress code on Fridays?",
        "Can I bring a dog to the Austin office?",
        "What is the internship return-offer bonus?",
        "Is there a pension plan besides the 401k?",
        "What is the parental leave policy for contractors?",
        "Does Acme cover quark therapy?",
    ]
    refusal = "I can't find that in the handbook."
    for question in unanswerable:
        for form in expand_question(question, []):
            push(form, refusal, [], "unanswerable", True)
    return questions


def to_documents(written: list[tuple[str, str, str]]) -> list[Document]:
    docs = []
    for doc_id, rel, text in written:
        meta_end = text.find("\n---", 3)
        raw = text[4:meta_end]
        body = text[meta_end + 4 :].strip()
        meta = yaml.safe_load(raw) or {}
        docs.append(
            Document(
                id=str(meta.get("id") or doc_id),
                title=str(meta.get("title") or doc_id),
                text=body,
                source=rel,
                metadata={key: value for key, value in meta.items() if key not in {"id", "title"}} | {"status": meta.get("status", "current")},
            )
        )
    return docs


def audit(documents: list[Document], questions: list[dict]) -> list[str]:
    config = default_config()
    chunks = chunk_documents(documents, config)
    catalog: list[tuple[str, str, str]] = []
    status = {doc.id: str(doc.metadata.get("status") or "current") for doc in documents}
    for chunk in chunks:
        for unit in units_from_text(chunk.text):
            catalog.append((chunk.doc_id, status.get(chunk.doc_id, "current"), unit))

    problems = []
    for item in questions:
        if item["unanswerable"]:
            worst = 0.0
            worst_unit = ""
            for _doc, _status, unit in catalog:
                score = term_overlap(item["question"], unit)
                if score > worst:
                    worst = score
                    worst_unit = unit
            if worst >= MIN_OVERLAP:
                problems.append(f"{item['id']} unanswerable overlap {worst:.2f}: {item['question']} || {worst_unit[:140]}")
            continue

        expected = [part.strip() for part in item["answer"].split(". ") if part.strip()]
        expected = [part if part.endswith(".") else part + "." for part in expected]
        # Linearized rows contain semicolons and a single period; don't split those.
        if item["type"] == "table":
            expected = [item["answer"]]

        candidates = []
        for doc_id, doc_status, unit in catalog:
            overlap = term_overlap(item["question"], unit)
            if doc_id in item["source_docs"]:
                rank = 0
            elif doc_status in {"superseded", "unofficial", "archived"}:
                rank = 1
            elif doc_status == "duplicate":
                rank = 2
            else:
                rank = 3
            candidates.append((overlap, overlap * rank_weight(rank), unit))
        candidates.sort(key=lambda row: (row[1], len(row[2])), reverse=True)
        selected = select_unit_texts(candidates)
        if set(selected) != set(expected):
            problems.append(
                f"{item['id']} {item['type']} selection mismatch\n  Q: {item['question']}\n  expected: {expected}\n  selected: {selected[:2]}"
            )
            if len(problems) > 25:
                break
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Acme handbook and gold eval set.")
    parser.add_argument("--out", default=str(ROOT / "data" / "acme"))
    args = parser.parse_args()
    out = Path(args.out)
    docs_root = out / "docs"
    if docs_root.exists():
        for stale in docs_root.rglob("*"):
            if stale.is_file():
                stale.unlink()

    fact_rows = facts()
    table_rows = tables()
    sections = build_sections(fact_rows)
    table_by_doc = {table["doc"]: table for table in table_rows}

    written: list[tuple[str, str, str]] = []
    for doc_id, meta in DOC_META.items():
        rel = meta[0]
        body = render_body(doc_id, sections, table_by_doc)
        text = dump_markdown(frontmatter(doc_id), body)
        written.append((doc_id, rel, text))

    bodies = {doc_id: text.split("---", 2)[-1] for doc_id, _rel, text in written}
    duplicates = [
        duplicate_of(
            "remote-work-2025",
            "remote-work-2025-copy",
            "Remote and Hybrid Work (IT copy)",
            "it/remote-work-2025-copy.md",
            bodies["remote-work-2025"],
        ),
        duplicate_of(
            "expense-limits",
            "expense-limits-quick-reference",
            "Expense Limits Quick Reference",
            "expenses/expense-limits-quick-reference.md",
            bodies["expense-limits"],
        ),
        duplicate_of(
            "health-plans",
            "health-plans-pocket-card",
            "Medical Plans Pocket Card",
            "benefits/health-plans-pocket-card.md",
            bodies["health-plans"],
        ),
    ]
    for rel, text in duplicates:
        doc_id = yaml.safe_load(text.split("---", 2)[1])["id"]
        written.append((doc_id, rel, text))

    written.append(("slack-ask-hr", "exports/slack-ask-hr.md", slack_hr()))
    written.append(("slack-ask-it", "exports/slack-ask-it.md", slack_it()))

    for _doc_id, rel, text in written:
        path = docs_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    questions = build_questions(fact_rows, table_rows)
    documents = to_documents(written)
    problems = audit(documents, questions)
    if problems:
        print(f"Audit failed with {len(problems)} issue(s):", file=sys.stderr)
        print("\n\n".join(problems[:12]), file=sys.stderr)
        raise SystemExit(1)

    eval_path = out / "eval.jsonl"
    eval_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8")
    suggested = [
        "How many PTO days after 3 years?",
        "What's the client dinner expense limit?",
        "How do I request parental leave?",
        "What is the lodging per diem in Chicago?",
        "Which health plan has an individual deductible of $500?",
        "What is the VPN gateway hostname?",
    ]
    (out / "suggested.json").write_text(json.dumps(suggested, indent=2) + "\n", encoding="utf-8")

    by_type: dict[str, int] = defaultdict(int)
    for row in questions:
        by_type[row["type"]] += 1
    print(f"Wrote {len(written)} documents to {docs_root}")
    print(f"Wrote {len(questions)} questions to {eval_path}")
    print("By type: " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())))
    if len(written) < 50 or len(questions) < 250:
        raise SystemExit(f"Expected >= 50 docs and >= 250 questions, got {len(written)} and {len(questions)}")


if __name__ == "__main__":
    main()
