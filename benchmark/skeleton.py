"""Pre-defined cascade skeletons for benchmark sample generation.

A skeleton declares a specific cascade scenario that a generated sample should
exhibit. The sampler picks a skeleton and injects its fields into the system
prompt, and the LLM produces a fitting multi-turn conversation plus a revision
instruction.

Each skeleton is rule-based and deterministic — the cascade produces
UNIQUELY-DETERMINED dependent values, with no convention or preference
required. The field names mirror the system-prompt slots:

* ``domain``: the domain this skeleton belongs to (e.g. ``travel_itinerary``),
  rendered verbatim into the prompt's Domain slot. Stamped automatically from
  the ``SKELETONS_BY_DOMAIN`` key each skeleton lives under.
* ``theme``: short natural-language label for the cascade scenario, rendered
  verbatim into the prompt's Theme slot
* ``conversation_pattern``: the dependency core to establish — the
  cascade-relevant RELATIONSHIPS and the ROLES of the fields, described without
  prescribing concrete key names, so the cascade fires deterministically while
  the LLM is free to choose the actual fields and shape. The surrounding
  artifact is left open for realistic, independent detail.
* ``edit_pattern``: which single element the revision instruction should change
* ``cascade_pattern``: how that edit propagates through the JSON, which values
  change vs stay, and why each new value is uniquely determined
* ``requires_master_data``: whether the cascade introduces a new entity whose
  attributes must be looked up. When true, the sample carries a separate
  ``master_data`` element (the attribute/lookup record) that is rendered into
  the edit instruction at evaluation time so the dependent values are uniquely
  determined rather than invented. master_data is an EDIT-PHASE concept only and
  is never mentioned in conversation_pattern/cascade_pattern.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Coarse taxonomy of the cascade mechanism a scenario exercises. Used as a
# stratification factor when splitting dev/test and for category-level analysis.
#   arithmetic   : numeric formula recompute chain (no dates / entity swap)
#   temporal     : date/time values shift along the dependency chain
#   substitution : an entity is swapped and its governed attributes are copied in
#   threshold    : effect is conditional on crossing / clamping to a threshold
#   add_remove   : a list element is added/removed and downstream relinks
#   status_flip  : a state change moves a set of status-governed fields together
Category = Literal["arithmetic", "temporal", "substitution", "threshold",
                   "add_remove", "status_flip"]


class Skeleton(BaseModel):
    domain: str = ""
    theme: str
    conversation_pattern: str
    edit_pattern: str
    cascade_pattern: str
    requires_master_data: bool = False
    category: Category | None = None


TRAVEL_ITINERARY: list[Skeleton] = [
    Skeleton(
        theme="Shifting the trip's anchor date",
        conversation_pattern="Center the artifact on a trip anchored on one date, where several other dates each derive from that anchor by a fixed offset stated in the conversation, so the anchor alone determines them. Around this core, add the realistic, independent details a traveler would record that do not depend on the dates, and vary their number from sample to sample.",
        edit_pattern="Move the anchor start date to a specific new date a few days from the original.",
        cascade_pattern="Every date that derives from the anchor shifts by the same delta, since each is the anchor plus a fixed offset, so the new anchor uniquely fixes them all. Values that do not derive from the anchor date stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Shifting the schedule's anchor time",
        conversation_pattern="Center the artifact on a single-day schedule anchored on one time, where several other times each derive from that anchor by a fixed offset stated in the conversation, so the anchor alone determines them. Around this core, add the realistic, independent details a traveler would record that do not depend on the times of day, and vary their number from sample to sample.",
        edit_pattern="Move the anchor time to a specific new time a fixed number of hours from the original.",
        cascade_pattern="Every time that derives from the anchor shifts by the same delta, since each is the anchor plus a fixed offset, so the new anchor fixes them all; a time crossing midnight rolls its date deterministically. Values that do not derive from the anchor time stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Changing the number of nights",
        conversation_pattern="Center the artifact on a trip with a start date and a stay length, where end dates and cumulative costs all derive from that length by stated rules (dates from the start plus the length, costs as per-unit rates times the length), so the length determines them. Around this core, add the realistic, independent details a traveler would record that do not depend on the stay length, and vary their number from sample to sample.",
        edit_pattern="Change the stay length (nights) to a specific new number.",
        cascade_pattern="End dates re-derive from the start plus the new length, and cumulative costs re-scale as the stated per-unit rate times the new length, so the new length fixes each. The start date and the per-unit rates stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Changing the party size",
        conversation_pattern="Center the artifact on a group trip with a party size, where per-unit quantities, aggregate totals, and per-person shares all derive from that count by stated arithmetic (the count times a unit price, sums of those, totals divided by the count), so the count determines them. Around this core, add the realistic, independent details a traveler would record that do not depend on the number of travelers, and vary their number from sample to sample.",
        edit_pattern="Change the party size to a specific new number of travelers.",
        cascade_pattern="Each quantity re-derives as the new count times its unit price, the aggregates re-sum, and per-person shares re-divide by the new count, so the new count fixes each. Unit prices and rates stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Swapping the order of two stops",
        conversation_pattern="Center the artifact on a one-day itinerary built as an ordered list of timed stops, where each stop's time follows from the previous stop's end plus a stated gap and its own stated duration, rolling up to overall counts and a day-end time. Around this core, add the realistic, independent details a planner would record that do not depend on a stop's position in the day, and vary their number and the number of stops from sample to sample.",
        edit_pattern="Swap the positions of two specific stops in the day.",
        cascade_pattern="When two stops exchange positions, the times from the earlier of the two positions onward recompute along the chain and any order-derived fields update; the counts and the day-end stay because the same stops and gaps are kept. Stops before the earlier position keep their times. This is determined because the durations, the gap, and the two positions are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Adding a stop to the day's schedule",
        conversation_pattern="Center the artifact on a one-day itinerary built as an ordered list of timed stops, where each stop's time follows from the previous stop's end plus a stated gap and its own stated duration, rolling up to overall counts and a day-end time. Around this core, add the realistic, independent details a planner would record that do not depend on a stop's position in the day, and vary their number and the number of stops from sample to sample.",
        edit_pattern="Add one specific sightseeing stop with a given visit duration at a specific point in the day.",
        cascade_pattern="When a stop is inserted, every later stop's time shifts later by the inserted stop's duration plus one gap, the day-end shifts later, and the stop count rises by one. Stops before the insertion point keep their times. This is determined because the durations and the gap are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Removing a stop from the day's schedule",
        conversation_pattern="Center the artifact on a one-day itinerary built as an ordered list of timed stops, where each stop's time follows from the previous stop's end plus a stated gap and its own stated duration, rolling up to overall counts and a day-end time. Around this core, add the realistic, independent details a planner would record that do not depend on a stop's position in the day, and vary their number and the number of stops from sample to sample.",
        edit_pattern="Remove one specific stop from the day.",
        cascade_pattern="When a stop is removed, every later stop's time shifts earlier by the removed stop's duration plus one gap, the day-end shifts earlier, and the stop count falls by one. Stops before the removed one keep their times. This is determined because the durations and the gap are stated.",
        requires_master_data=False,
    ),
]


INVOICE: list[Skeleton] = [
    Skeleton(
        theme="Recomputing the invoice total chain",
        conversation_pattern="Center the artifact on an invoice whose totals form a chain, where each line item is labeled with what it is for and its amount derives from a quantity and a unit price, a subtotal sums the line amounts, and charges and a final balance build on the subtotal, so the inputs determine the whole chain. Each line item must be distinguishable from the others by its label, so a single line can be named unambiguously. Around this core, add the realistic, independent details an invoice would carry that do not depend on the line amounts, and vary their number from sample to sample.",
        edit_pattern="Change one line item's unit price or quantity to a specific new value.",
        cascade_pattern="Editing one input re-derives its line amount, then the chain recomputes level by level: the subtotal re-sums and every total built on it follows in order. Other lines and the stated rates stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Prorating an unused subscription period",
        conversation_pattern="Center the artifact on a subscription charge with a fee, an explicitly stated billing-period length in days, and used and unused day spans, where a prorated credit and the net charge follow from the fee and the unused span by standard daily proration, so the fee and the spans determine them. Choose the fee and the period length so the daily rate divides evenly, keeping the credit exact. Around this core, add the realistic, independent details a billing record would carry that do not depend on the fee or the unused span, and vary their number from sample to sample.",
        edit_pattern="Change the monthly fee (or the cancellation date that sets the unused days) to a specific new value.",
        cascade_pattern="Editing the fee (or the span that sets the unused days) re-derives the prorated credit and the net charge by standard daily proration, and any totals built on them follow. The period length and the spans stay. Each new value is fixed because daily proration on the stated period has one unambiguous result.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Marking the invoice as paid",
        conversation_pattern="Center the artifact on an invoice currently in one status, carrying a set of fields whose present values follow from that status, where moving it to the paid status has one well-understood effect on each. Around this core, add the realistic, independent details an invoice would carry that do not depend on the status, and vary their number from sample to sample.",
        edit_pattern="Change the invoice status from issued to paid.",
        cascade_pattern="Marking it paid moves every status-governed field to its paid-state value, while fields the status does not govern stay. Each new value is fixed because the effect of payment on each field is unambiguous.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Recording a partial payment",
        conversation_pattern="Center the artifact on an invoice carrying a total, an amount paid, an outstanding balance that is the total minus the amount paid, a status that follows from whether the balance has reached zero, and a running paid-to-date that totals the payments, so the amount paid determines them. Around this core, add the realistic, independent details an invoice would carry that do not depend on the amount paid, and vary their number from sample to sample.",
        edit_pattern="Change the amount paid to a specific new value.",
        cascade_pattern="Editing the amount paid re-derives the outstanding balance, moves the status to its value for the new balance (settled once the balance reaches zero, otherwise still partially settled), and re-totals the paid-to-date. The total stays. Each new value is fixed because the effect of the amount paid on the balance and status is unambiguous.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Propagating a changed invoice number",
        conversation_pattern="Center the artifact on an invoice that declares one identifier, where several other fields embed or reference that identifier as a visible token. Around this core, add the realistic, independent details an invoice would carry that do not embed that identifier, and vary their number from sample to sample.",
        edit_pattern="Change the invoice number to a different identifier.",
        cascade_pattern="Changing the identifier updates every field that embeds it, substituting the token while the surrounding text stays. Identifiers and fields that do not reference it are unaffected.",
        requires_master_data=False,
    ),
]


SHOPPING_CART: list[Skeleton] = [
    Skeleton(
        theme="Substituting one product for another",
        conversation_pattern="Center the artifact on a cart holding a product line whose attributes are stated on the line — including the numeric ones that drive money or weight totals — with line and cart totals and any fees derived from those attributes by stated arithmetic, so the line's values determine them. Around this core, add the realistic, independent details a cart would carry that do not depend on that product line, and vary their number from sample to sample.",
        edit_pattern="Replace one product in the cart with a different specific product.",
        cascade_pattern="Every attribute of the swapped line takes the replacement product's value (the replacement's attributes are supplied with the edit instruction), and every total or fee derived from those attributes recomputes. Other lines and unrelated fields stay. This is determined because the replacement's attributes are provided with the edit.",
        requires_master_data=True,
    ),
    Skeleton(
        theme="Swapping the base product and its compatible accessories",
        conversation_pattern="Center the artifact on a cart with one base product and several accessories whose choice depends on that base, each being the one valid for it. Around this core, add the realistic, independent details a cart would carry that do not depend on the base product, and vary their number from sample to sample.",
        edit_pattern="Change the base product (device, platform, or model) to a different specific one.",
        cascade_pattern="Each base-dependent accessory switches to the one valid for the new base (the valid accessories for the new base are supplied with the edit instruction), while accessories and fields that do not depend on the base stay. This is determined because the valid accessories for the new base are provided with the edit.",
        requires_master_data=True,
    ),
    Skeleton(
        theme="Scaling a product bundle's components",
        conversation_pattern="Center the artifact on a cart with a bundle carrying a quantity and its component lines, where each component's quantity and amount derive from the bundle quantity by stated arithmetic (a per-bundle count times the bundle quantity, times a unit price), rolling up to cart totals. Around this core, add the realistic, independent details a cart would carry that do not depend on the bundle quantity, and vary their number from sample to sample.",
        edit_pattern="Change the bundle quantity to a specific new number.",
        cascade_pattern="Editing the bundle quantity re-derives every component's quantity and amount from the stated per-bundle counts and prices, and the cart totals follow. The per-bundle counts and unit prices stay. This is determined because each component's per-bundle count and price are stated in the conversation.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Removing a product and its bound accessories",
        conversation_pattern="Center the artifact on a cart with a main product plus accessories that each belong to one of the main products, where each accessory's tie to its main is made evident from its own name or description rather than from a dedicated link field naming the main it depends on, rolling up to a count and totals. Around this core, add the realistic, independent details a cart would carry — including products and accessories that belong to a different main — that do not depend on the removed main, and vary their number from sample to sample.",
        edit_pattern="Remove one main product, along with the accessories that go with it, from the cart. Name the main product and refer to its accompanying accessories collectively, without listing them individually.",
        cascade_pattern="Removing a main product also removes every accessory that belongs to it, and the count and totals re-aggregate. Accessories that belong to a different main stay. This is determined because each accessory's main is unambiguous from its name or description.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Switching the order to gift mode",
        conversation_pattern="Center the artifact on a cart that is currently a non-gift order, carrying a set of presentation fields whose present values follow from it not being a gift, where switching it to a gift order flips each to a single, unambiguous gift-state value — booleans or fixed states only, never free-form text or an open-ended choice. Around this core, add the realistic, independent details a cart would carry that do not depend on whether the order is a gift, and vary their number from sample to sample.",
        edit_pattern="Turn gift mode on for the order.",
        cascade_pattern="Switching to a gift order moves every gift-governed presentation field to its gift-state value, while fields the gift setting does not govern stay. Each new value is fixed because the effect of the gift setting on each field is unambiguous.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Crossing the free-shipping threshold",
        conversation_pattern="Center the artifact on a cart whose subtotal sums its line amounts, carrying a stated free-shipping threshold and a stated flat shipping fee that applies only while the subtotal is below the threshold and drops to zero once the subtotal reaches it, with a total that is the subtotal plus the shipping fee, so the line amounts determine the subtotal, the shipping fee, and the total. State the threshold and the flat fee in the conversation, and set the initial subtotal on one side of the threshold so the shipping fee is in its corresponding state. Around this core, add the realistic, independent details a cart would carry that do not depend on the subtotal, and vary their number from sample to sample.",
        edit_pattern="Change one line item's quantity or unit price to a specific new value chosen so the subtotal moves across the stated free-shipping threshold.",
        cascade_pattern="Editing the line re-derives its amount and then the subtotal; whether the new subtotal reaches the stated threshold determines the shipping fee (zero at or above the threshold, the stated flat fee below it), and the total re-sums the subtotal plus the shipping fee. Lines and fields that do not feed the subtotal stay. This is determined because the threshold and the flat fee are stated in the conversation.",
        requires_master_data=False,
    ),
]


PROJECT_SCHEDULE: list[Skeleton] = [
    Skeleton(
        theme="Changing a task's duration",
        conversation_pattern="Center the artifact on a task list where each task has a stated duration and starts when its predecessor finishes, so finishes and start dates derive along the precedence chain, rolling up to a project end. Around this core, add the realistic, independent details a project plan would carry that do not depend on the task durations or dates, and vary their number and the number of tasks from sample to sample.",
        edit_pattern="Change one task's duration to a specific new number of days.",
        cascade_pattern="Editing a task's duration re-derives its finish, then each downstream task's start and finish recompute along the chain to the project end. Tasks not downstream of the edit stay. This is determined because durations and links are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Inserting a task into the schedule",
        conversation_pattern="Center the artifact on a task list where each task has a stated duration and starts when its predecessor finishes, so finishes derive along the precedence chain, rolling up to a task count and a project end. Around this core, add the realistic, independent details a project plan would carry that do not depend on the schedule dates, and vary their number and the number of tasks from sample to sample.",
        edit_pattern="Insert one new task with a specific duration at a specific point in the precedence chain.",
        cascade_pattern="Inserting a task shifts every downstream task later by the inserted duration, raises the task count by one, and moves the project end. Tasks before the insertion point stay. This is determined because durations and links are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Removing a task from the schedule",
        conversation_pattern="Center the artifact on a task list where each task has a stated duration and names the predecessor it starts after, so finishes derive along the precedence chain and a removed task's successors can relink, rolling up to a task count and a project end. Around this core, add the realistic, independent details a project plan would carry that do not depend on the schedule dates, and vary their number and the number of tasks from sample to sample.",
        edit_pattern="Remove one task from the middle of the precedence chain.",
        cascade_pattern="Removing a task relinks its successors to its predecessor, pulls downstream dates earlier, lowers the task count by one, and moves the project end. Tasks upstream of the removal stay. This is determined because durations and links are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Repointing a task's predecessor",
        conversation_pattern="Center the artifact on a task list where each task has a stated duration and starts when its predecessor finishes, so finishes and the project end derive from the precedence links. Around this core, add the realistic, independent details a project plan would carry that do not depend on the dependency links, and vary their number and the number of tasks from sample to sample.",
        edit_pattern="Change one task's predecessor to a different specific task.",
        cascade_pattern="Repointing a task to a new predecessor re-derives its start from that predecessor's finish, then its finish and all downstream dates recompute, and the project end follows. Tasks not affected by the relink stay. This is determined because finishes and the new link are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Moving a milestone date",
        conversation_pattern="Center the artifact on a schedule with a milestone several tasks must finish before, where each feeding task's slack derives from the milestone and that task's finish. Around this core, add the realistic, independent details a project plan would carry that do not depend on the milestone date, and vary their number from sample to sample.",
        edit_pattern="Move the milestone (or deadline) to a specific new date.",
        cascade_pattern="Moving the milestone re-derives each feeding task's slack, and its latest allowable start, from the new milestone date. Tasks that do not feed the milestone stay. This is determined because the feeding links and finishes are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Reassigning a task to another owner",
        conversation_pattern="Center the artifact on a task list where each task has an owner and an effort estimate, with a per-owner workload aggregating each owner's tasks and a capacity check derived from a stated capacity. Around this core, add the realistic, independent details a project plan would carry that do not depend on who owns each task, and vary their number and the number of tasks and owners from sample to sample.",
        edit_pattern="Reassign one task from its current owner to a different specific owner.",
        cascade_pattern="Reassigning a task moves its effort from the old owner's workload to the new owner's, re-sums both, and re-evaluates each owner's capacity check against the stated capacity. Other tasks and owners stay. This is determined because the efforts and capacities are stated.",
        requires_master_data=False,
    ),
]


COURSE_PLAN: list[Skeleton] = [
    Skeleton(
        theme="Changing a course grade",
        conversation_pattern="Center the artifact on a course list where each course has credit hours and a letter grade, with a stated letter-grade-to-grade-point scale from which each course's grade points derive, a GPA derives as the credit-weighted average of those grade points, and standing flags derive from the GPA against stated thresholds. Around this core, add the realistic, independent details a course plan would carry that do not depend on the grades, and vary their number and the number of courses from sample to sample.",
        edit_pattern="Change one course's letter grade to a specific new letter grade.",
        cascade_pattern="Editing a course's letter grade re-derives that course's grade points from the stated scale, then the GPA re-derives as the credit-weighted average, and the standing flags re-evaluate against the stated thresholds. Other grades and credits stay. This is determined because the grade scale and thresholds are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Changing a course's credit hours",
        conversation_pattern="Center the artifact on a course list where each course has credit hours and grade points, a GPA derives as the credit-weighted average, and credit totals and a remaining-to-requirement derive from the credits against a stated requirement. Around this core, add the realistic, independent details a course plan would carry that do not depend on the credit hours, and vary their number and the number of courses from sample to sample.",
        edit_pattern="Change one course's credit hours to a specific new number.",
        cascade_pattern="Editing a course's credits re-derives the GPA with its new weight, re-sums the credit total, and re-derives the remaining-to-requirement. Other courses and the requirement stay. This is determined because grades, credits, and the requirement are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Dropping a course",
        conversation_pattern="Center the artifact on a course list where each course carries credit hours and grade points, a credit-weighted GPA derives from them, and a running credit total derives from the credits, so the courses together determine both. Around this core, add the realistic, independent details a course plan would carry that do not depend on which courses are in the list, and vary their number and the number of courses from sample to sample.",
        edit_pattern="Drop one specific course from the plan.",
        cascade_pattern="Dropping a course removes its credit hours and grade points from the list, so the credit total drops by its credits and the GPA re-derives as the credit-weighted average of the remaining courses. Other courses and the requirement stay. This is determined because every remaining course's credits and grade points are stated.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Adding a course",
        conversation_pattern="Center the artifact on a course list where each course carries credit hours, with a running credit total and a remaining-to-requirement derived from those credits against a stated requirement, so the credits determine them. Around this core, add the realistic, independent details a course plan would carry that do not depend on the credit total, and vary their number and the number of courses from sample to sample.",
        edit_pattern="Add one specific course, with its credit hours, to the plan.",
        cascade_pattern="Adding the course raises the credit total by its credit hours and lowers the remaining-to-requirement by the same amount. Other courses and the requirement stay. This is determined because the added course's credit hours are given with the edit.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Changing a course's prerequisite",
        conversation_pattern="Center the artifact on a course list where each planned course's eligibility derives from whether its prerequisite chain is satisfied by a stated set of completed courses, with the eligibility rule stated. Around this core, add the realistic, independent details a course plan would carry that do not depend on the prerequisites, and vary their number and the number of courses from sample to sample.",
        edit_pattern="Change one course's prerequisite to a different specific course.",
        cascade_pattern="Changing a course's prerequisite to a new one (whose own prerequisite chain is supplied with the edit instruction) re-derives that course's eligibility from whether the new chain is satisfied by the completed set. Courses with unchanged prerequisites stay. This is determined because the new prerequisite's chain is provided with the edit.",
        requires_master_data=True,
    ),
    Skeleton(
        theme="Changing the degree credit requirement",
        conversation_pattern="Center the artifact on a degree plan with a credit requirement and credits earned, where a remaining-to-requirement, an on-track flag, and a projected completion derive from the requirement, the earned credits, and a stated per-term load. Around this core, add the realistic, independent details a degree plan would carry that do not depend on the credit requirement, and vary their number from sample to sample.",
        edit_pattern="Change the degree's required total credits to a specific new number.",
        cascade_pattern="Editing the requirement re-derives the remaining-to-requirement, re-evaluates the on-track flag, and re-derives the projected completion from the remaining and the stated per-term load. The earned credits and per-term load stay. This is determined because the per-term load and earned credits are stated.",
        requires_master_data=False,
    ),
]


DATA_PIPELINE: list[Skeleton] = [
    # M1 arithmetic recompute
    Skeleton(
        theme="Changing the pipeline schedule interval",
        conversation_pattern="Center the artifact on a scheduled pipeline whose run interval determines freshness SLA, retry window, downstream expected refresh time, and stale-data alert threshold by arithmetic rules written explicitly in the artifact (state each formula, e.g. freshness_sla = interval + 15). Around this core, add the realistic, independent details an operations record would carry that do not depend on the schedule interval, and vary their number from sample to sample.",
        edit_pattern="Change the pipeline schedule interval to a different specific interval.",
        cascade_pattern="Editing the interval re-derives the SLA, retry window, downstream refresh expectation, and stale-data alert threshold from the stated formulas. Owners, descriptions, and fields that do not read the interval stay unchanged.",
        requires_master_data=False,
    ),
    # M3 entity-attribute inheritance (swap via master_data)
    Skeleton(
        theme="Switching the source connector",
        conversation_pattern="Center the artifact on a pipeline source connector whose connector attributes feed extraction path, credential reference, owner group, ingest mode, and stated cost or latency estimates. Around this core, add the realistic, independent details a data pipeline spec would carry that do not depend on the connector, and vary their number from sample to sample. Do not embed a lookup or reference card for any connector inside the artifact; the artifact holds only the current connector's own fields.",
        edit_pattern="Switch the source connector to a different specific connector, and supply the replacement connector's attributes in master_data.",
        cascade_pattern="Every connector-governed field takes the replacement connector's supplied attributes, and derived estimates recompute from those attributes. Downstream transforms that do not read connector attributes stay unchanged. This is determined because the replacement connector's attributes are provided with the edit.",
        requires_master_data=True,
    ),
    # M6 conditional / threshold-gated propagation
    Skeleton(
        theme="Crossing a data-volume threshold",
        conversation_pattern="Center the artifact on a dataset with a stated daily row-volume figure, a volume-derived storage-size estimate (stated formula), and a STATED volume threshold rule with two explicit branches: if daily volume exceeds the threshold, the partition strategy must be 'partitioned' and the compaction tier and retention class step up to specific stated targets; if it is at or below the threshold, the strategy stays 'single_file' and those tiers stay at their stated baseline. Record the threshold and both branch targets explicitly so the outcome is uniquely determined. Around this core, add realistic independent details that do not depend on volume, and vary their number from sample to sample.",
        edit_pattern="Change the daily row volume to a specific new number.",
        cascade_pattern="Editing the volume re-derives the storage-size estimate from the stated formula, AND — only if the new volume crosses the stated threshold relative to the old value — flips the partition strategy and steps the compaction tier and retention class to the stated branch targets. If the threshold is not crossed, the strategy and tiers stay unchanged. The outcome is determined by the stated threshold and branch targets.",
        requires_master_data=False,
    ),
    # M4 reference / token propagation (the one canonical recall test)
    Skeleton(
        theme="Renaming an upstream table",
        conversation_pattern="Center the artifact on a data pipeline DAG where one upstream table name is referenced by downstream SQL, lineage entries, dashboard source references, and alert queries as a visible token. Around this core, add the realistic, independent details a data engineering runbook would carry that do not reference that table name, including other table names that must NOT be touched, and vary their number from sample to sample.",
        edit_pattern="Rename the upstream table to a different specific table identifier.",
        cascade_pattern="Changing the table identifier substitutes the new token in every SQL statement, lineage reference, dashboard source, and alert query that references the old name, while preserving surrounding text and leaving every other table reference unchanged. This is determined because each reference embeds the old table name visibly.",
        requires_master_data=False,
    ),
    # M7 constraint change with conditional invalidation
    Skeleton(
        theme="Tightening the freshness-SLA policy",
        conversation_pattern="Center the artifact on a pipeline with several downstream consumers, each carrying a refresh-interval-minutes value and a compliance flag, governed by a stated policy field 'max_allowed_staleness_minutes' with an explicit rule: a consumer is compliant iff its refresh interval is at most the policy max; a non-compliant consumer must have its refresh interval clamped down to the policy max and its flag set to 'non_compliant'. Make some consumers compliant and some not under the current policy, and ensure every consumer's initial compliance flag already matches the stated rule under the current policy (a consumer within the current max is compliant). Around this core, add realistic independent details that do not depend on the policy, and vary the number of consumers from sample to sample.",
        edit_pattern="Tighten the policy's max_allowed_staleness_minutes to a specific stricter (smaller) value.",
        cascade_pattern="Tightening the policy clamps the refresh interval of only those consumers whose interval now exceeds the new max down to that max and flips their flag to 'non_compliant'; consumers already within the new max and all unrelated fields stay unchanged. The affected set is determined by comparing each consumer's stated interval to the new policy max.",
        requires_master_data=False,
    ),
]


SOFTWARE_DEPLOYMENT_CONFIG: list[Skeleton] = [
    # M1 arithmetic recompute (every derived field carries an explicit stated formula)
    Skeleton(
        theme="Changing the replica count",
        conversation_pattern="Center the artifact on a deployment with a replica count, per-pod CPU and memory requests, and availability settings, where EVERY derived field carries its own explicit formula written in the artifact: total CPU = replicas * perPodCpu, total memory = replicas * perPodMem, capacity summary = replicas, and each disruption-budget value has a stated formula (e.g. minAvailable = replicas - 1, maxUnavailable = 1). Do not include any availability field whose dependence on the replica count is not given by a stated formula. Around this core, add the realistic, independent details a deployment config would carry that do not depend on the replica count, and vary their number from sample to sample.",
        edit_pattern="Change the replica count to a specific new number.",
        cascade_pattern="Editing the replica count re-derives total CPU, total memory, capacity summary, and each disruption-budget value strictly from its stated per-pod value and formula. Per-pod requests and unrelated deployment fields stay unchanged.",
        requires_master_data=False,
    ),
    # M3 entity-attribute inheritance (swap via master_data)
    Skeleton(
        theme="Switching the deployment environment",
        conversation_pattern="Center the artifact on a service configured for one environment, where namespace, hostnames, secret references, alert routing, and release channel all follow from the environment's supplied attributes. Around this core, add the realistic, independent details a deployment record would carry that do not depend on the environment, and vary their number from sample to sample. Do not embed a lookup or reference card for any environment inside the artifact; the artifact holds only the current environment's own fields.",
        edit_pattern="Move the deployment to a different specific environment, and supply the target environment's attributes in master_data.",
        cascade_pattern="Every environment-governed field takes the replacement environment's supplied attributes: namespace, hostnames, secret references, alert routing, and release channel. Service-specific fields that do not depend on the environment stay unchanged. This is determined because the replacement environment's attributes are provided with the edit.",
        requires_master_data=True,
    ),
    # M6 conditional / threshold-gated propagation
    Skeleton(
        theme="Crossing the node memory budget",
        conversation_pattern="Center the artifact on a deployment with a replica count, a per-pod memory request, a stated total-memory formula (total = replicas * perPodMem), and a STATED node memory budget with two explicit branches: if total memory exceeds the budget, 'node_pool_tier' steps up to a specific stated larger tier and a 'needs_larger_nodes' flag is true; if total is at or below the budget, the tier stays at its stated baseline and the flag is false. Record the budget and both branch targets explicitly. Around this core, add realistic independent details that do not depend on memory, and vary their number from sample to sample.",
        edit_pattern="Change the per-pod memory request to a specific new value.",
        cascade_pattern="Editing the per-pod memory re-derives total memory from the stated formula, AND — only if total memory now crosses the stated node budget relative to before — steps node_pool_tier to the stated target and flips needs_larger_nodes. If the budget boundary is not crossed, the tier and flag stay unchanged. The outcome is determined by the stated budget and branch targets.",
        requires_master_data=False,
    ),
    # M5 removal + reaggregation
    Skeleton(
        theme="Removing a sidecar container",
        conversation_pattern="Center the artifact on a pod with several containers, each with a name, a CPU request, a memory request, and an exposed port, where the pod's container_count, total_cpu, total_memory (stated sums over the containers), and exposed_port_list aggregate over the included containers. Around this core, add the realistic, independent details a deployment record would carry that do not depend on the container list, and vary the number of containers from sample to sample.",
        edit_pattern="Remove one specific sidecar container from the pod.",
        cascade_pattern="Removing the container deletes it from the container list, lowers container_count, removes its port from exposed_port_list, and re-derives total_cpu and total_memory as the stated sums over the remaining containers. Unrelated deployment metadata stays unchanged.",
        requires_master_data=False,
    ),
    # M7 constraint change with conditional invalidation
    Skeleton(
        theme="Tightening the per-container memory quota",
        conversation_pattern="Center the artifact on a deployment with several containers, each carrying a memory_request_mib value and an over_quota flag, governed by a stated policy field 'per_container_memory_quota_mib' with an explicit rule: a container is compliant iff its request is at most the quota; a container exceeding the quota must have its request clamped down to the quota and its over_quota flag set true. Make some containers compliant and some not under the current quota, and ensure every container's initial over_quota flag already matches the stated rule under the current quota (a container within the current quota is not over quota). Around this core, add realistic independent details that do not depend on the quota, and vary the number of containers from sample to sample.",
        edit_pattern="Tighten the per_container_memory_quota_mib to a specific stricter (smaller) value.",
        cascade_pattern="Tightening the quota clamps the memory request of only those containers whose request now exceeds the new quota down to that quota and sets their over_quota flag true; containers already within the new quota and all unrelated fields stay unchanged. The affected set is determined by comparing each container's stated request to the new quota.",
        requires_master_data=False,
    ),
]


ORG_ACCESS_PLAN: list[Skeleton] = [
    # M3 entity-attribute inheritance (the one canonical swap)
    Skeleton(
        theme="Changing an employee role",
        conversation_pattern="Center the artifact on an employee access plan where the employee's role determines permission groups, approval limit, required trainings, review interval, and support queue by role attributes. Around this core, add the realistic, independent details an access record would carry that do not depend on the role, and vary their number from sample to sample. Do not embed a lookup or reference card for any role inside the artifact; the artifact holds only the current role's own fields.",
        edit_pattern="Change the employee's role to a different specific role, and supply the replacement role's attributes in master_data.",
        cascade_pattern="Every role-governed field takes the replacement role's supplied attributes, including permission groups, approval limit, required trainings, review interval, and support queue. Identity fields and unrelated assignments stay unchanged. This is determined because the replacement role's attributes are provided with the edit.",
        requires_master_data=True,
    ),
    # M2 temporal / transitive-chain recompute
    Skeleton(
        theme="Reassigning a manager",
        conversation_pattern="Center the artifact on an employee record where the manager determines a transitive approval path, reporting chain, escalation contact, and team roll-up count. Provide the replacement manager's own reporting attributes (chain, approval path, escalation contact, and team roll-up count) so each dependent value is uniquely determined: the new reporting chain is the employee prepended to the manager's supplied chain, the new approval path is the employee prepended to the manager's supplied approval path, the escalation contact is the manager's supplied escalation contact, and the team roll-up count is set to the manager's supplied team roll-up count. Around this core, add the realistic, independent details an org access plan would carry that do not depend on the manager, and vary their number from sample to sample. Do not embed a lookup or reference card for any manager inside the artifact; the artifact holds only the current manager's own reporting fields.",
        edit_pattern="Change the employee's manager to a different specific manager, and supply that manager's reporting attributes in master_data.",
        cascade_pattern="Changing the manager re-derives the approval path and reporting chain as the employee prepended to the replacement manager's supplied path and chain, sets the escalation contact to the replacement manager's supplied escalation contact, and sets the team roll-up count to the replacement manager's supplied team roll-up count. Role, access tier, and unrelated fields stay unchanged.",
        requires_master_data=True,
    ),
    # M6 conditional / threshold-gated propagation
    Skeleton(
        theme="Crossing the elevated-approval threshold",
        conversation_pattern="Center the artifact on an access plan with an approval_limit_usd and a STATED threshold rule with two explicit branches: if the approval limit exceeds the threshold, an extra approval step (a specific stated approver, e.g. 'VP co-approval') is appended to the approval steps, mfa_required becomes true, and review_cadence shortens to a specific stated stricter value; if the limit is at or below the threshold, the approval steps, mfa, and cadence stay at their stated baseline. Record the threshold and both branch targets explicitly. Around this core, add realistic independent details that do not depend on the limit, and vary their number from sample to sample.",
        edit_pattern="Change the approval_limit_usd to a specific new amount.",
        cascade_pattern="Editing the approval limit updates it, AND — only if the new amount crosses the stated threshold relative to before — appends the stated extra approval step, sets mfa_required true, and shortens review_cadence to the stated value. If the threshold is not crossed, those fields stay unchanged. The outcome is determined by the stated threshold and branch targets.",
        requires_master_data=False,
    ),
    # M5 removal + reaggregation
    Skeleton(
        theme="Revoking a system access",
        conversation_pattern="Center the artifact on an access plan with a list of granted systems (each with a name and a sensitivity level), an access_count, and a capability summary of boolean flags (e.g. can_export_data, can_admin) each derived by a stated rule over the granted systems (e.g. can_admin is true iff any granted system has sensitivity 'high'). Around this core, add the realistic, independent details an access record would carry that do not depend on the grant list, and vary the number of granted systems from sample to sample.",
        edit_pattern="Revoke one specific granted system from the access plan.",
        cascade_pattern="Revoking the system removes it from the grant list, lowers access_count, and re-derives every capability-summary flag from its stated rule over the remaining grants (a flag flips only if the removed system was what satisfied it). Identity and unrelated fields stay unchanged.",
        requires_master_data=False,
    ),
    # M7 constraint change with conditional invalidation
    Skeleton(
        theme="Tightening the session-timeout policy",
        conversation_pattern="Center the artifact on an access plan listing several access entries, each with a session_timeout_minutes value and a compliance flag, governed by a stated policy field 'max_session_timeout_minutes' with an explicit rule: an entry is compliant iff its timeout is at most the policy max; an entry exceeding it must be clamped down to the max and flagged 'non_compliant'. Make some entries compliant and some not under the current policy, and ensure every entry's initial compliance flag already matches the stated rule under the current policy (an entry within the current max is compliant). Around this core, add realistic independent details that do not depend on the policy, and vary the number of entries from sample to sample.",
        edit_pattern="Tighten the max_session_timeout_minutes to a specific stricter (smaller) value.",
        cascade_pattern="Tightening the policy clamps the timeout of only those entries whose timeout now exceeds the new max down to that max and flips their flag to 'non_compliant'; entries already within the new max and all unrelated fields stay unchanged. The affected set is determined by comparing each entry's stated timeout to the new policy max.",
        requires_master_data=False,
    ),
]


MANUFACTURING_BOM: list[Skeleton] = [
    Skeleton(
        theme="Changing the build quantity",
        conversation_pattern="Center the artifact on a manufacturing build plan with a finished-unit quantity, component requirements per finished unit, labor per unit, and per-unit costs, where total component demand, labor hours, batch cost, and procurement quantities derive from the build quantity by stated arithmetic. Around this core, add the realistic, independent details a build plan would carry that do not depend on build quantity, and vary their number from sample to sample.",
        edit_pattern="Change the build quantity to a specific new number of finished units.",
        cascade_pattern="Editing the build quantity re-derives every component demand, procurement quantity, labor total, and batch cost from the stated per-unit requirements and costs. Per-unit requirements, suppliers, and unrelated fields stay unchanged.",
        requires_master_data=False,
    ),
    # M6 conditional / threshold-gated propagation
    Skeleton(
        theme="Crossing the shipping-weight class",
        conversation_pattern="Center the artifact on a BOM where one component's per-unit quantity drives its weight contribution and the assembly total weight, defined explicitly as a stated sum over component weight contributions, and a STATED shipping-class rule with two explicit branches: if the assembly total weight exceeds a stated threshold, 'shipping_class' steps up to a specific stated heavier class and the packaging spec upgrades to a stated target; if total weight is at or below the threshold, the class and packaging stay at their stated baseline. Record the threshold and both branch targets explicitly. Around this core, add the realistic, independent details a BOM would carry that do not depend on that component quantity, and vary their number from sample to sample.",
        edit_pattern="Change one component's per-unit quantity to a specific new value.",
        cascade_pattern="Editing the component's per-unit quantity re-derives its total demand, component cost, and weight contribution, and the assembly total weight as the stated sum, AND — only if the total weight now crosses the stated threshold relative to before — steps shipping_class to the stated heavier class and upgrades the packaging spec. If the threshold is not crossed, the class and packaging stay unchanged. Other component quantities and unrelated fields stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Substituting a component supplier",
        conversation_pattern="Center the artifact on a BOM component sourced from one supplier, where supplier attributes determine unit cost, lead time, country-of-origin, risk flag, and procurement contact. Around this core, add the realistic, independent details a manufacturing plan would carry that do not depend on the supplier, and vary their number from sample to sample. Do not embed a lookup or reference card for any supplier inside the artifact; the artifact holds only the current supplier's own fields.",
        edit_pattern="Switch one component to a different specific supplier.",
        cascade_pattern="Every supplier-governed field takes the replacement supplier's supplied attributes, and derived totals such as cost and lead time recompute from them. Component quantity and unrelated components stay unchanged. This is determined because the replacement supplier's attributes are provided with the edit.",
        requires_master_data=True,
    ),
    Skeleton(
        theme="Changing the scrap rate",
        conversation_pattern="Center the artifact on a manufacturing plan with a stated scrap rate, usable-unit target, raw-unit requirement, waste estimate, and cost totals where raw demand derives from the target and scrap rate by stated formulas. Around this core, add the realistic, independent details a production record would carry that do not depend on scrap rate, and vary their number from sample to sample.",
        edit_pattern="Change the scrap rate to a specific new percentage.",
        cascade_pattern="Editing the scrap rate re-derives raw units required, waste estimate, and any cost totals that depend on raw demand from the stated formulas. The usable-unit target and per-unit costs stay unchanged.",
        requires_master_data=False,
    ),
    Skeleton(
        theme="Removing a component from the assembly",
        conversation_pattern="Center the artifact on a BOM assembly with several components, where component count, total material cost, total weight, supplier list, and procurement summary aggregate over the included components. Around this core, add the realistic, independent details a BOM record would carry that do not depend on the component list, and vary their number and the number of components from sample to sample.",
        edit_pattern="Remove one specific component from the assembly.",
        cascade_pattern="Removing the component deletes it from the component list, lowers the component count, and re-aggregates material cost, weight, supplier list, and procurement summary over the remaining components. Unrelated assembly metadata stays unchanged.",
        requires_master_data=False,
    ),
]


SKELETONS_BY_DOMAIN: dict[str, list[Skeleton]] = {
    "travel_itinerary": TRAVEL_ITINERARY,
    "invoice": INVOICE,
    "shopping_cart": SHOPPING_CART,
    "project_schedule": PROJECT_SCHEDULE,
    "course_plan": COURSE_PLAN,
    "data_pipeline": DATA_PIPELINE,
    "software_deployment_config": SOFTWARE_DEPLOYMENT_CONFIG,
    "org_access_plan": ORG_ACCESS_PLAN,
    "manufacturing_bom": MANUFACTURING_BOM,
}

# Stamp each skeleton with its domain (the dict key it lives under), so the
# domain can be injected into the prompt directly from the Skeleton.
for _domain, _skeletons in SKELETONS_BY_DOMAIN.items():
    for _sk in _skeletons:
        _sk.domain = _domain

# Cascade-mechanism category per scenario (keyed by theme, which is unique across
# the registry). Hand-assigned from each scenario's cascade_pattern; kept in one
# block so the labels are auditable and editable. Stamped onto each skeleton.
_CATEGORY_BY_THEME: dict[str, Category] = {
    # travel_itinerary
    "Shifting the trip's anchor date": "temporal",
    "Shifting the schedule's anchor time": "temporal",
    "Changing the number of nights": "temporal",
    "Changing the party size": "arithmetic",
    "Swapping the order of two stops": "temporal",
    "Adding a stop to the day's schedule": "add_remove",
    "Removing a stop from the day's schedule": "add_remove",
    # invoice
    "Recomputing the invoice total chain": "arithmetic",
    "Prorating an unused subscription period": "arithmetic",
    "Marking the invoice as paid": "status_flip",
    "Recording a partial payment": "arithmetic",
    "Propagating a changed invoice number": "substitution",
    # shopping_cart
    "Substituting one product for another": "substitution",
    "Swapping the base product and its compatible accessories": "substitution",
    "Scaling a product bundle's components": "arithmetic",
    "Removing a product and its bound accessories": "add_remove",
    "Switching the order to gift mode": "status_flip",
    "Crossing the free-shipping threshold": "threshold",
    # project_schedule
    "Changing a task's duration": "temporal",
    "Inserting a task into the schedule": "add_remove",
    "Removing a task from the schedule": "add_remove",
    "Repointing a task's predecessor": "temporal",
    "Moving a milestone date": "temporal",
    "Reassigning a task to another owner": "substitution",
    # course_plan
    "Changing a course grade": "arithmetic",
    "Changing a course's credit hours": "arithmetic",
    "Dropping a course": "add_remove",
    "Adding a course": "add_remove",
    "Changing a course's prerequisite": "substitution",
    "Changing the degree credit requirement": "arithmetic",
    # data_pipeline
    "Changing the pipeline schedule interval": "arithmetic",
    "Switching the source connector": "substitution",
    "Crossing a data-volume threshold": "threshold",
    "Renaming an upstream table": "substitution",
    "Tightening the freshness-SLA policy": "threshold",
    # software_deployment_config
    "Changing the replica count": "arithmetic",
    "Switching the deployment environment": "substitution",
    "Crossing the node memory budget": "threshold",
    "Removing a sidecar container": "add_remove",
    "Tightening the per-container memory quota": "threshold",
    # org_access_plan
    "Changing an employee role": "substitution",
    "Reassigning a manager": "substitution",
    "Crossing the elevated-approval threshold": "threshold",
    "Revoking a system access": "add_remove",
    "Tightening the session-timeout policy": "threshold",
    # manufacturing_bom
    "Changing the build quantity": "arithmetic",
    "Crossing the shipping-weight class": "threshold",
    "Substituting a component supplier": "substitution",
    "Changing the scrap rate": "arithmetic",
    "Removing a component from the assembly": "add_remove",
}
for _skeletons in SKELETONS_BY_DOMAIN.values():
    for _sk in _skeletons:
        _sk.category = _CATEGORY_BY_THEME[_sk.theme]


def skeleton_at(index: int) -> Skeleton:
    """Domain-interleaved round-robin selection.

    The first round emits the first skeleton of each domain, the second round
    emits the second skeleton of each domain, and so on. Within a domain the
    skeletons wrap once exhausted, so domains shorter than the longest one
    simply repeat earlier themes on later rounds."""
    domains = list(SKELETONS_BY_DOMAIN.values())
    skels = domains[index % len(domains)]
    round_idx = index // len(domains)
    return skels[round_idx % len(skels)]
