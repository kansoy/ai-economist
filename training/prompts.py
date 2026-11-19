DRAFT_SYSTEM = "You are an economist and a careful writer."

DRAFT_USER = (
    "Rewrite the paragraph below in your own words, the way you would naturally write it for an economics paper. "
    "Keep every piece of content: every fact, number, year, citation, name, example and qualification, and add nothing. "
    "Reply with the paragraph only.\n\n{paragraph}"
)

WRITER_USER = (
    "Journal: {journal} ({journal_family}). Section: {section}.\n"
    "End of the previous paragraph: {prev_tail}\n\n"
    "Revise this draft paragraph into the authors' own published prose. "
    "Keep every fact, number and citation.\n\n"
    "Draft:\n{draft}"
)

INTERNAL_JUDGE_USER = (
    "Below are two versions of one paragraph from {where}. "
    "One was written by the paper's authors and published. "
    "The other was produced by a revision tool from a draft. "
    "Which one did the authors write? Answer with the single letter A or B.\n\n"
    "Paragraph A:\n{a}\n\nParagraph B:\n{b}"
)

OUTSIDE_JUDGE_SYSTEM = "Answer with one letter only."

OUTSIDE_JUDGE_USER = (
    "Below are two versions of one paragraph from the introduction of an economics paper. "
    "One was written by the paper's authors. The other was written by an AI model. "
    "Which one did the authors write? Answer with the single letter A or B.\n\n"
    "Paper: {title}\n\nParagraph A:\n{a}\n\nParagraph B:\n{b}"
)

FIDELITY_SYSTEM = (
    "You check whether an edited paragraph keeps the content of a source paragraph. "
    "You are careful and literal. Reply with JSON only."
)

FIDELITY_USER = (
    "Below is a SOURCE paragraph from an economics paper and an EDITED version of it. "
    "The edited version was supposed to keep every piece of content of the source: "
    "every claim, number, year, name, citation, example and qualification (hedge), and to add nothing. "
    "Wording, order and sentence structure may change freely.\n\n"
    "List every place where the content differs, using these categories:\n"
    "- dropped: a claim, example or detail of the source that the edited version leaves out;\n"
    "- added: a claim, example or detail in the edited version that the source does not contain;\n"
    "- hedge: a qualification whose strength changes (for example \"may\" becomes \"does\", "
    "\"some\" becomes \"most\", a caveat disappears);\n"
    "- entity: a name, place, institution, dataset, citation or number that is missing, changed or attributed differently;\n"
    "- meaning: a claim that is still present but now says something different "
    "(for example a causal claim made descriptive, or the reverse).\n"
    "Do not list differences of wording that keep the meaning.\n\n"
    "Return one JSON object: {\"dropped\": [...], \"added\": [...], \"hedge\": [...], "
    "\"entity\": [...], \"meaning\": [...], \"verdict\": \"same\" or \"minor\" or \"major\"}. "
    "Each list item is a short quote or description.\n"
    "Verdict \"same\": no content difference. "
    "\"minor\": only differences a careful author would accept as the same content "
    "(a slightly weaker or stronger word, a detail made a little more or less specific without changing the claim). "
    "\"major\": any dropped or added claim, any changed number, name or citation, "
    "any changed hedge that alters what is claimed, or any change of meaning.\n\n"
    "SOURCE:\n{src}\n\nEDITED:\n{ed}"
)
