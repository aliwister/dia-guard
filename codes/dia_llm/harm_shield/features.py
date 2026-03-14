# features.py - Complete feature definitions for 235 linguistic features across 12 grammatical categories
# Based on eWAVE (Electronic World Atlas of Varieties of English) - https://ewave-atlas.org

FEATURE_LIBRARY = {
    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 1: PRONOUNS (eWAVE Features 1-47)
    # Pronoun exchange, nominal gender, case, reflexives, possessives, address forms
    # ═══════════════════════════════════════════════════════════════════════════

    "she_inanimate": {
        "id": 1,
        "category": "pronouns",
        "description": "She/her used for inanimate referents (ships, cars, countries)",
        "examples": [
            ("The car broke down. It needs repair.", "The car broke down. She needs repair."),
            ("Look at that ship. It is huge.", "Look at that ship. She is huge."),
            ("She was burning good.", "She was burning good.")  # about a house
        ]
    },

    "he_inanimate": {
        "id": 2,
        "category": "pronouns",
        "description": "He/him used for inanimate referents",
        "examples": [
            ("I bet you can't climb it.", "I bet thee cansn't climb he."),  # referring to a tree
            ("The sun is bright.", "He is bright.")
        ]
    },

    "alternative_it_referential": {
        "id": 3,
        "category": "pronouns",
        "description": "Alternative forms/phrases for referential (non-dummy) it, like 'the thing'",
        "examples": [
            ("Switch it off.", "Off the thing."),
            ("When you switch it off, press that button.", "When you off the thing, you press that one.")
        ]
    },

    "alternative_it_dummy": {
        "id": 4,
        "category": "pronouns",
        "description": "Alternative forms/phrases for dummy it in weather/impersonal expressions",
        "examples": [
            ("It's raining.", "Thass rainen."),
            ("It is cold.", "Cold it is.")
        ]
    },

    "generalized_3sg_subject": {
        "id": 5,
        "category": "pronouns",
        "description": "Generalized third person singular pronoun for all genders in subject position",
        "examples": [
            ("He/she went home.", "'Em went home."),
            ("He is tall. She is short.", "Em tall. Em short.")
        ]
    },

    "generalized_3sg_object": {
        "id": 6,
        "category": "pronouns",
        "description": "Generalized third person singular pronoun for all genders in object position",
        "examples": [
            ("I saw him/her.", "I saw om."),
            ("Give it to them.", "Give it to 'em.")
        ]
    },

    "me_coordinate_subjects": {
        "id": 7,
        "category": "pronouns",
        "description": "Me instead of I in coordinate subjects",
        "examples": [
            ("John and I went to the store.", "Me and John went to the store."),
            ("She and I are friends.", "Me and her are friends."),
            ("My brother and I were late.", "Me and my brother were late.")
        ]
    },

    "myself_coordinate_subjects": {
        "id": 8,
        "category": "pronouns",
        "description": "Myself/meself instead of I in coordinate subjects",
        "examples": [
            ("My husband and I were late.", "My husband and myself were late."),
            ("John and I did it.", "John and meself did it.")
        ]
    },

    "benefactive_dative": {
        "id": 9,
        "category": "pronouns",
        "description": "Benefactive 'personal dative' construction with object pronoun",
        "examples": [
            ("I got a new car.", "I got me a new car."),
            ("She bought a dress.", "She bought her a dress.")
        ]
    },

    "no_gender_distinction": {
        "id": 10,
        "category": "pronouns",
        "description": "No gender distinction in third person singular pronouns",
        "examples": [
            ("My mother, she is a teacher.", "My mother, he's a teacher."),
            ("My father, he works here.", "My father, she works here.")
        ]
    },

    "regularized_reflexives": {
        "id": 11,
        "category": "pronouns",
        "description": "Regularized reflexives paradigm (hisself, theirselves)",
        "examples": [
            ("He hurt himself.", "He hurt hisself."),
            ("They did it themselves.", "They did it theirselves."),
            ("She helped herself.", "She helped herownself.")
        ]
    },

    "object_pronoun_reflexive_base": {
        "id": 12,
        "category": "pronouns",
        "description": "Object pronoun forms serving as base for reflexives",
        "examples": [
            ("I hurt myself.", "I hurt meself."),
            ("We did it ourselves.", "We did it usselves.")
        ]
    },

    "subject_pronoun_reflexive_base": {
        "id": 13,
        "category": "pronouns",
        "description": "Subject pronoun forms serving as base for reflexives",
        "examples": [
            ("They helped themselves.", "They helped theyselves."),
            ("We did it ourselves.", "We did it weselves.")
        ]
    },

    "no_number_reflexives": {
        "id": 14,
        "category": "pronouns",
        "description": "No number distinction in reflexives",
        "examples": [
            ("We did it ourselves.", "We did it ourself."),
            ("They helped themselves.", "They helped themself.")
        ]
    },

    "absolute_reflexives": {
        "id": 15,
        "category": "pronouns",
        "description": "Absolute use of reflexives (independent, referring to important person)",
        "examples": [
            ("The boss has gone to Dublin.", "Himself is gone to Dublin."),
            ("The lady of the house is out.", "Herself is out.")
        ]
    },

    "emphatic_reflexives_own": {
        "id": 16,
        "category": "pronouns",
        "description": "Emphatic reflexives with 'own'",
        "examples": [
            ("Everybody took care of themselves.", "Everybody took care of their own self."),
            ("I did it myself.", "I did it my own self.")
        ]
    },

    "fi_possessive": {
        "id": 17,
        "category": "pronouns",
        "description": "Creation of possessive pronouns with prefix fi- + personal pronoun",
        "examples": [
            ("That's my job.", "Das fi-me work."),
            ("It's his book.", "A fi-him book.")
        ]
    },

    "subject_possessive_1sg": {
        "id": 18,
        "category": "pronouns",
        "description": "Subject pronoun forms as possessive: first person singular (I for my)",
        "examples": [
            ("My book is here.", "I book is here.")
        ]
    },

    "subject_possessive_1pl": {
        "id": 19,
        "category": "pronouns",
        "description": "Subject pronoun forms as possessive: first person plural (we for our)",
        "examples": [
            ("Our farm is big.", "We farm is big.")
        ]
    },

    "subject_possessive_3sg": {
        "id": 20,
        "category": "pronouns",
        "description": "Subject pronoun forms as possessive: third person singular (he for his)",
        "examples": [
            ("His book is here.", "He book is here.")
        ]
    },

    "subject_possessive_3pl": {
        "id": 21,
        "category": "pronouns",
        "description": "Subject pronoun forms as possessive: third person plural (they for their)",
        "examples": [
            ("It's their book.", "It's they book.")
        ]
    },

    "you_possessive": {
        "id": 22,
        "category": "pronouns",
        "description": "You as (modifying) possessive pronoun",
        "examples": [
            ("Your fare is expensive.", "You fare is expensive.")
        ]
    },

    "alternative_2p_possessive": {
        "id": 23,
        "category": "pronouns",
        "description": "Second person pronoun forms other than 'you' as possessive",
        "examples": [
            ("Your eyes are beautiful.", "Unu ai is beautiful.")
        ]
    },

    "object_possessive_3sg": {
        "id": 24,
        "category": "pronouns",
        "description": "Object pronoun forms as possessive: third person singular",
        "examples": [
            ("His dog is big.", "Im dog is big."),
            ("Her book is here.", "Her book is here.")
        ]
    },

    "object_possessive_3pl": {
        "id": 25,
        "category": "pronouns",
        "description": "Object pronoun forms as possessive: third person plural",
        "examples": [
            ("Their book is here.", "Them book is here.")
        ]
    },

    "object_possessive_1sg": {
        "id": 26,
        "category": "pronouns",
        "description": "Object pronoun forms as possessive: first person singular",
        "examples": [
            ("He's my brother.", "He's me brother.")
        ]
    },

    "object_possessive_1pl": {
        "id": 27,
        "category": "pronouns",
        "description": "Object pronoun forms as possessive: first person plural",
        "examples": [
            ("Our George was a nice one.", "Us George was a nice one.")
        ]
    },

    "us_np_subject": {
        "id": 28,
        "category": "pronouns",
        "description": "Use of us + NP in subject function",
        "examples": [
            ("We kids used to steal sweets.", "Us kids used to pinch the sweets.")
        ]
    },

    "us_singular_object": {
        "id": 29,
        "category": "pronouns",
        "description": "Use of us in object function with singular referent (= me)",
        "examples": [
            ("Show me those boots.", "Show us them boots."),
            ("Give me that.", "Give us that.")
        ]
    },

    "subject_pronoun_object": {
        "id": 30,
        "category": "pronouns",
        "description": "Non-coordinated subject pronoun forms in object function",
        "examples": [
            ("You did get him out of bed.", "You did get he out of bed."),
            ("I saw her.", "I saw she.")
        ]
    },

    "object_pronoun_subject": {
        "id": 31,
        "category": "pronouns",
        "description": "Non-coordinated object pronoun forms in subject function",
        "examples": [
            ("What did they call it?", "What did 'em call it?"),
            ("She went home.", "Her went home.")
        ]
    },

    "emphatic_vs_nonemphatic": {
        "id": 32,
        "category": "pronouns",
        "description": "Distinction between emphatic vs. non-emphatic forms of pronouns",
        "examples": [
            ("I did it. (emphatic)", "Mi did it. (non-emphatic) / A did it. (emphatic)")
        ]
    },

    "nasal_possessive": {
        "id": 33,
        "category": "pronouns",
        "description": "Independent possessive pronoun forms with added nasal (hisn, hern, ourn)",
        "examples": [
            ("That book is his.", "That book is hisn."),
            ("This car is theirs.", "This car is theirn."),
            ("The house is ours.", "The house is ourn.")
        ]
    },

    "second_person_plural": {
        "id": 34,
        "category": "pronouns",
        "description": "Forms or phrases for second person plural other than you (y'all, youse, you guys)",
        "examples": [
            ("Are you all coming?", "Are y'all coming?"),
            ("You guys should leave.", "Youse should leave."),
            ("Do you all want some?", "Do y'all want some?")
        ]
    },

    "second_person_singular": {
        "id": 35,
        "category": "pronouns",
        "description": "Forms or phrases for second person singular other than you (thou, ye)",
        "examples": [
            ("You should go.", "Ye should go."),
            ("Do you want some?", "Dost thou want some?")
        ]
    },

    "inclusive_exclusive_1p": {
        "id": 36,
        "category": "pronouns",
        "description": "Distinct forms for inclusive/exclusive first person non-singular",
        "examples": [
            ("We (including you) will go.", "Yumi go."),
            ("We (excluding you) will go.", "Mifela go.")
        ]
    },

    "number_distinctions_pronouns": {
        "id": 37,
        "category": "pronouns",
        "description": "More number distinctions than singular vs. plural (dual, trial)",
        "examples": [
            ("We two will go.", "Tufala go."),
            ("We three will go.", "Trifala go.")
        ]
    },

    "specialized_plural_markers": {
        "id": 38,
        "category": "pronouns",
        "description": "Specialized plural markers for pronouns",
        "examples": [
            ("We (group) are here.", "Us-gang are here."),
            ("You guys are coming.", "Yu gaiz coming.")
        ]
    },

    "plural_interrogative_elements": {
        "id": 39,
        "category": "pronouns",
        "description": "Plural forms of interrogative pronouns using additional elements",
        "examples": [
            ("Who will be there?", "Who-all will be there?"),
            ("What do you want?", "What-all do you want?")
        ]
    },

    "plural_interrogative_reduplication": {
        "id": 40,
        "category": "pronouns",
        "description": "Plural forms of interrogative pronouns through reduplication",
        "examples": [
            ("Who came? (multiple people)", "Who-who came?")
        ]
    },

    "singular_it_plural": {
        "id": 41,
        "category": "pronouns",
        "description": "Singular 'it' for plural 'they' in anaphoric use",
        "examples": [
            ("Those books... They can be obtained.", "Those books... It can be obtained.")
        ]
    },

    "object_pronoun_drop": {
        "id": 42,
        "category": "pronouns",
        "description": "Object pronoun drop",
        "examples": [
            ("I like it.", "Mi laekem."),
            ("Give it to me.", "Give to me.")
        ]
    },

    "subject_drop_referential": {
        "id": 43,
        "category": "pronouns",
        "description": "Subject pronoun drop: referential pronouns",
        "examples": [
            ("I sold them already.", "Sold already."),
            ("He went home.", "Went home.")
        ]
    },

    "subject_drop_dummy": {
        "id": 44,
        "category": "pronouns",
        "description": "Subject pronoun drop: dummy pronouns",
        "examples": [
            ("It rained yesterday.", "Rained yesterday."),
            ("It is cold.", "Cold.")
        ]
    },

    "insertion_it": {
        "id": 45,
        "category": "pronouns",
        "description": "Insertion of 'it' where Standard English favors zero",
        "examples": [
            ("As I made clear before...", "As I made it clear before...")
        ]
    },

    "deletion_it_referential": {
        "id": 46,
        "category": "pronouns",
        "description": "Deletion of 'it' in referential it is-constructions",
        "examples": [
            ("It is very nice food.", "Is very nice food.")
        ]
    },

    "deletion_it_nonreferential": {
        "id": 47,
        "category": "pronouns",
        "description": "Deletion of 'it' in non-referential it is-constructions",
        "examples": [
            ("It is not allowed to stop here.", "Here is not allowed to stop the car.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 2: NOUN PHRASE (eWAVE Features 48-87)
    # Plurals, articles, determiners, possession, comparison
    # ═══════════════════════════════════════════════════════════════════════════

    "regularized_plurals_s": {
        "id": 48,
        "category": "noun_phrase",
        "description": "Regularization of plural formation: extension of -s to irregular plurals",
        "examples": [
            ("The children played.", "The childrens played."),
            ("I saw two oxen.", "I saw two oxens."),
            ("The sheep are grazing.", "The sheeps are grazing.")
        ]
    },

    "regularized_plurals_phonological": {
        "id": 49,
        "category": "noun_phrase",
        "description": "Regularization of plural formation: phonological regularization",
        "examples": [
            ("The wives are here.", "The wifes are here."),
            ("Many knives were found.", "Many knifes were found."),
            ("The leaves fell.", "The leafs fell.")
        ]
    },

    "plural_preposed": {
        "id": 50,
        "category": "noun_phrase",
        "description": "Plural marking via preposed elements",
        "examples": [
            ("The boys are here.", "Olketa boe are here.")
        ]
    },

    "plural_postposed": {
        "id": 51,
        "category": "noun_phrase",
        "description": "Plural marking via postposed elements",
        "examples": [
            ("The women are here.", "Woman dem are here.")
        ]
    },

    "associative_plural_them": {
        "id": 52,
        "category": "noun_phrase",
        "description": "Associative plural marked by postposed 'and them/them all/dem'",
        "examples": [
            ("My dad and his friends/family.", "My dad and them."),
            ("John and his group.", "John dem.")
        ]
    },

    "associative_plural_other": {
        "id": 53,
        "category": "noun_phrase",
        "description": "Associative plural marked by other elements",
        "examples": [
            ("My dad and his associates.", "My Daddy gang.")
        ]
    },

    "group_plurals": {
        "id": 54,
        "category": "noun_phrase",
        "description": "Group plurals (plural on head noun in compound titles)",
        "examples": [
            ("Two Secretaries of State.", "Two Secretary of States.")
        ]
    },

    "mass_noun_plurals": {
        "id": 55,
        "category": "noun_phrase",
        "description": "Different count/mass distinctions resulting in plural for StE singular",
        "examples": [
            ("The wood is here.", "The woods is here."),
            ("I need advice.", "I need advices."),
            ("The staff is ready.", "The staffs are ready.")
        ]
    },

    "zero_plural_quantifier": {
        "id": 56,
        "category": "noun_phrase",
        "description": "Absence of plural marking only after quantifiers",
        "examples": [
            ("Four pounds.", "Four pound."),
            ("Five years old.", "Five year old."),
            ("Twenty miles away.", "Twenty mile away.")
        ]
    },

    "zero_plural_human": {
        "id": 57,
        "category": "noun_phrase",
        "description": "Plural marking generally optional for nouns with human referents",
        "examples": [
            ("My sisters are pretty girls.", "My sister are pretty girl.")
        ]
    },

    "zero_plural_nonhuman": {
        "id": 58,
        "category": "noun_phrase",
        "description": "Plural marking generally optional for nouns with non-human referents",
        "examples": [
            ("The trees don't grow tall.", "The tree don't grow very tall."),
            ("I have two books.", "I have two book.")
        ]
    },

    "double_determiners": {
        "id": 59,
        "category": "noun_phrase",
        "description": "Double determiners",
        "examples": [
            ("This problem is serious.", "This our common problem is very serious."),
            ("That house is old.", "That there house is old.")
        ]
    },

    "definite_for_indefinite": {
        "id": 60,
        "category": "noun_phrase",
        "description": "Use of definite article where StE has indefinite article",
        "examples": [
            ("I had a toothache.", "I had the toothache.")
        ]
    },

    "indefinite_for_definite": {
        "id": 61,
        "category": "noun_phrase",
        "description": "Use of indefinite article where StE has definite article",
        "examples": [
            ("The sun was shining.", "A sun was shining.")
        ]
    },

    "zero_for_definite": {
        "id": 62,
        "category": "noun_phrase",
        "description": "Use of zero article where StE has definite article",
        "examples": [
            ("Did you get the mileage claim?", "Did you get mileage-claim?"),
            ("I went to the hospital.", "I went to hospital.")
        ]
    },

    "zero_for_indefinite": {
        "id": 63,
        "category": "noun_phrase",
        "description": "Use of zero article where StE has indefinite article",
        "examples": [
            ("Getting a girl from India.", "Getting girl from India."),
            ("She is a teacher.", "She is teacher.")
        ]
    },

    "definite_for_zero": {
        "id": 64,
        "category": "noun_phrase",
        "description": "Use of definite article where StE favors zero",
        "examples": [
            ("At Nestle Ghana Ltd.", "At the Nestle, Ghana Ltd.")
        ]
    },

    "indefinite_for_zero": {
        "id": 65,
        "category": "noun_phrase",
        "description": "Use of indefinite article where StE favors zero",
        "examples": [
            ("About three fields.", "About a three fields.")
        ]
    },

    "indefinite_one": {
        "id": 66,
        "category": "noun_phrase",
        "description": "Indefinite article one/wan",
        "examples": [
            ("They saw a green snake.", "They seen one green snake.")
        ]
    },

    "demonstrative_for_definite": {
        "id": 67,
        "category": "noun_phrase",
        "description": "Demonstratives for definite articles",
        "examples": [
            ("The door closed.", "That door bin close.")
        ]
    },

    "them_for_those": {
        "id": 68,
        "category": "noun_phrase",
        "description": "Them instead of demonstrative those",
        "examples": [
            ("In those days.", "In them days."),
            ("One of those things.", "One of them things.")
        ]
    },

    "yon_yonder": {
        "id": 69,
        "category": "noun_phrase",
        "description": "Yon/yonder indicating remoteness",
        "examples": [
            ("That oil company over there.", "Yon oil company.")
        ]
    },

    "demonstrative_here_there": {
        "id": 70,
        "category": "noun_phrase",
        "description": "Proximal and distal demonstratives with 'here' and 'there'",
        "examples": [
            ("This book vs. those books.", "This here book vs. them there books.")
        ]
    },

    "no_number_demonstratives": {
        "id": 71,
        "category": "noun_phrase",
        "description": "No number distinction in demonstratives",
        "examples": [
            ("I've watched these children.", "I've watched this children.")
        ]
    },

    "group_genitives": {
        "id": 72,
        "category": "noun_phrase",
        "description": "Group genitives (possessive on entire phrase)",
        "examples": [
            ("The man I met's girlfriend.", "The man I met's girlfriend.")
        ]
    },

    "existential_possessive": {
        "id": 73,
        "category": "noun_phrase",
        "description": "Existential construction to express possessive",
        "examples": [
            ("I have a car.", "Ma moto de.")
        ]
    },

    "for_possessive_post": {
        "id": 74,
        "category": "noun_phrase",
        "description": "Phrases with for + noun to express possession: for-phrase following possessed NP",
        "examples": [
            ("Chinyere's knife.", "Nayf for Chinyere.")
        ]
    },

    "for_possessive_pre": {
        "id": 75,
        "category": "noun_phrase",
        "description": "Phrases with for + noun to express possession: for-phrase preceding possessed NP",
        "examples": [
            ("My sister's husband.", "He was for my sister husband.")
        ]
    },

    "bilong_possessive": {
        "id": 76,
        "category": "noun_phrase",
        "description": "Postnominal phrases with bilong/blong/long/blo to express possession",
        "examples": [
            ("The man's dog.", "Dog blong maan.")
        ]
    },

    "zero_genitive": {
        "id": 77,
        "category": "noun_phrase",
        "description": "Omission of genitive suffix; possession through bare juxtaposition",
        "examples": [
            ("My daddy's brother.", "My daddy brother."),
            ("John's car is red.", "John car is red.")
        ]
    },

    "double_comparatives": {
        "id": 78,
        "category": "noun_phrase",
        "description": "Double comparatives and superlatives",
        "examples": [
            ("It's much easier.", "It's so much more easier."),
            ("He is the biggest.", "He is the most biggest.")
        ]
    },

    "synthetic_comparison": {
        "id": 79,
        "category": "noun_phrase",
        "description": "Regularized comparison: extension of synthetic marking (-er, -est)",
        "examples": [
            ("The most regular kind of guy.", "The regularest kind of guy."),
            ("More beautiful.", "Beautifuler.")
        ]
    },

    "analytic_comparison": {
        "id": 80,
        "category": "noun_phrase",
        "description": "Regularized comparison: extension of analytic marking (more, most)",
        "examples": [
            ("One of the prettiest sunsets.", "One of the most pretty sunsets.")
        ]
    },

    "much_comparative": {
        "id": 81,
        "category": "noun_phrase",
        "description": "Much as comparative marker",
        "examples": [
            ("More severe than in Singapore.", "Much severe than in Singapore.")
        ]
    },

    "as_to_comparative": {
        "id": 82,
        "category": "noun_phrase",
        "description": "As/to as comparative markers",
        "examples": [
            ("Worse than before.", "Worse as before.")
        ]
    },

    "participle_comparatives": {
        "id": 83,
        "category": "noun_phrase",
        "description": "Comparatives and superlatives of participles",
        "examples": [
            ("The most fighting person.", "The fightingest person."),
            ("The best singer.", "The singingest one.")
        ]
    },

    "than_only_comparative": {
        "id": 84,
        "category": "noun_phrase",
        "description": "Comparative marking only with 'than' (no more/-er)",
        "examples": [
            ("He loves his car more than his children.", "He loves his car than his children.")
        ]
    },

    "more_and_comparative": {
        "id": 85,
        "category": "noun_phrase",
        "description": "Comparative marking with more...and",
        "examples": [
            ("More powder on their hands and faces.", "More powder on their hands and in their faces.")
        ]
    },

    "zero_degree": {
        "id": 86,
        "category": "noun_phrase",
        "description": "Zero marking of degree",
        "examples": [
            ("One of the most radical students.", "One of the radical students.")
        ]
    },

    "postnominal_adjectives": {
        "id": 87,
        "category": "noun_phrase",
        "description": "Attributive adjectival modifiers follow head noun",
        "examples": [
            ("A big iron saucepan.", "Bikpela sospen ain.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 3: TENSE AND ASPECT (eWAVE Features 88-113)
    # Progressive, habitual, perfect, future, tense markers
    # ═══════════════════════════════════════════════════════════════════════════

    "progressive_stative": {
        "id": 88,
        "category": "tense_aspect",
        "description": "Wider range of progressive be + V-ing: extension to stative verbs",
        "examples": [
            ("I understand the problem.", "I am understanding the problem."),
            ("She knows the answer.", "She is knowing the answer."),
            ("What do you want?", "What are you wanting?")
        ]
    },

    "progressive_habitual": {
        "id": 89,
        "category": "tense_aspect",
        "description": "Wider range of progressive be + V-ing: extension to habitual contexts",
        "examples": [
            ("I usually go to the library.", "I am usually going to library.")
        ]
    },

    "invariant_be_habitual": {
        "id": 90,
        "category": "tense_aspect",
        "description": "Invariant be as habitual marker",
        "examples": [
            ("He is always sick.", "He be sick."),
            ("She is usually working.", "She be working."),
            ("They are often here.", "They be here.")
        ],
        "notes": "Indicates habitual or repeated action, NOT simple present"
    },

    "do_habitual": {
        "id": 91,
        "category": "tense_aspect",
        "description": "Do as habitual marker",
        "examples": [
            ("He catches fish often.", "He does catch fish pretty.")
        ]
    },

    "other_habitual_synthetic": {
        "id": 92,
        "category": "tense_aspect",
        "description": "Other non-standard habitual markers: synthetic",
        "examples": [
            ("I drink three or four cups at a meal.", "I drinks three and four cups to a meal.")
        ]
    },

    "other_habitual_analytic": {
        "id": 93,
        "category": "tense_aspect",
        "description": "Other non-standard habitual markers: analytic",
        "examples": [
            ("He is often sick.", "He do be sick a lot."),
            ("I keep running.", "Me stap ronron.")
        ]
    },

    "stap_stay_progressive": {
        "id": 94,
        "category": "tense_aspect",
        "description": "Progressive marker stap or stay",
        "examples": [
            ("He is eating.", "Hem i stap kaekae."),
            ("I am working.", "I stay working.")
        ]
    },

    "be_sat_stood": {
        "id": 95,
        "category": "tense_aspect",
        "description": "Be sat/stood with progressive meaning",
        "examples": [
            ("When you are standing there.", "When you're stood there you can see the flames.")
        ]
    },

    "there_resultative": {
        "id": 96,
        "category": "tense_aspect",
        "description": "There with past participle in resultative contexts",
        "examples": [
            ("Something has fallen down the sink.", "There's something fallen down the sink.")
        ]
    },

    "medial_object_perfect": {
        "id": 97,
        "category": "tense_aspect",
        "description": "Medial object perfect (object between auxiliary and participle)",
        "examples": [
            ("You have made the stations.", "You have the stations made.")
        ]
    },

    "after_perfect": {
        "id": 98,
        "category": "tense_aspect",
        "description": "After-perfect (after + V-ing for recent completion)",
        "examples": [
            ("She has just sold the boat.", "She's after selling the boat."),
            ("I have just finished eating.", "I'm after eating.")
        ]
    },

    "simple_past_for_perfect": {
        "id": 99,
        "category": "tense_aspect",
        "description": "Simple past for StE present perfect",
        "examples": [
            ("Have you ever been to London?", "Were you ever in London?"),
            ("I have seen that movie.", "I saw that movie.")
        ]
    },

    "perfect_for_simple_past": {
        "id": 100,
        "category": "tense_aspect",
        "description": "Present perfect for StE simple past",
        "examples": [
            ("Some of us went to New York years ago.", "Some of us have been to New York years ago.")
        ]
    },

    "simple_present_for_perfect": {
        "id": 101,
        "category": "tense_aspect",
        "description": "Simple present for continuative or experiential perfect",
        "examples": [
            ("I have known her since she was a child.", "I know her since she was a child.")
        ]
    },

    "be_perfect_auxiliary": {
        "id": 102,
        "category": "tense_aspect",
        "description": "Be as perfect auxiliary (with motion/change verbs)",
        "examples": [
            ("They haven't left school yet.", "They're not left school yet."),
            ("I have gone home.", "I am gone home.")
        ]
    },

    "do_tense_marker": {
        "id": 103,
        "category": "tense_aspect",
        "description": "Do as unstressed tense marker",
        "examples": [
            ("This man who owns this.", "This man what do own this.")
        ]
    },

    "completive_done": {
        "id": 104,
        "category": "tense_aspect",
        "description": "Completive/perfect done",
        "examples": [
            ("He has gone fishing.", "He done go fishing."),
            ("I have finished.", "I done finished."),
            ("She has eaten.", "She done ate.")
        ]
    },

    "have_done_completive": {
        "id": 105,
        "category": "tense_aspect",
        "description": "Completive/perfect have/be + done + past participle",
        "examples": [
            ("He has gone.", "He is done gone.")
        ]
    },

    "be_done_irrealis": {
        "id": 106,
        "category": "tense_aspect",
        "description": "Sequential or irrealis be done",
        "examples": [
            ("They will have eaten you alive.", "They be done eat you alive.")
        ]
    },

    "completive_slam": {
        "id": 107,
        "category": "tense_aspect",
        "description": "Completive/perfect marker slam",
        "examples": [
            ("I already told you not to mess up.", "I slam told you not to mess up.")
        ]
    },

    "ever_experiential": {
        "id": 108,
        "category": "tense_aspect",
        "description": "Ever as marker of experiential perfect",
        "examples": [
            ("I have seen the movie.", "I ever see the movie.")
        ]
    },

    "already_perfect": {
        "id": 109,
        "category": "tense_aspect",
        "description": "Perfect marker already",
        "examples": [
            ("We moved here a week ago.", "We did move here a week already."),
            ("I have eaten.", "I eat already.")
        ]
    },

    "finish_completive": {
        "id": 110,
        "category": "tense_aspect",
        "description": "Finish-derived completive markers",
        "examples": [
            ("We have finished working in our garden.", "Wakum gaden blong mifala finis.")
        ]
    },

    "been_past_anterior": {
        "id": 111,
        "category": "tense_aspect",
        "description": "Past tense/anterior marker been (remote past BIN in AAVE)",
        "examples": [
            ("I cut the bread.", "I been cut the bread."),
            ("I have known him for years.", "I been knowing him.")
        ]
    },

    "had_bare_root": {
        "id": 112,
        "category": "tense_aspect",
        "description": "Anterior had + bare root",
        "examples": [
            ("He had eaten bread before he went to school.", "Hii had iit do bred biifoh hii goo tuu skuul.")
        ]
    },

    "sequence_tenses_loosening": {
        "id": 113,
        "category": "tense_aspect",
        "description": "Loosening of sequence of tenses rule",
        "examples": [
            ("I noticed the van I had come in.", "I noticed the van I came in.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 4: MODAL VERBS (eWAVE Features 114-127)
    # Future markers, double modals, quasi-modals
    # ═══════════════════════════════════════════════════════════════════════════

    "go_future": {
        "id": 114,
        "category": "modal_verbs",
        "description": "Go-based future markers (gonna, gon)",
        "examples": [
            ("He will build my house.", "He gon build my house."),
            ("I am going to eat.", "I'm gon eat.")
        ]
    },

    "volition_future": {
        "id": 115,
        "category": "modal_verbs",
        "description": "Volition-based future markers other than will",
        "examples": [
            ("I want to go.", "I wan go.")
        ]
    },

    "come_future": {
        "id": 116,
        "category": "modal_verbs",
        "description": "Come-based future/ingressive markers",
        "examples": [
            ("I am about to cook your meal.", "I am coming to cook your meal.")
        ]
    },

    "present_neutral_future": {
        "id": 117,
        "category": "modal_verbs",
        "description": "Present tense forms for neutral future reference",
        "examples": [
            ("I think I will make a new dress for Chinese New Year.", "I think I make one new dress for Chinese New Year.")
        ]
    },

    "is_for_am_will": {
        "id": 118,
        "category": "modal_verbs",
        "description": "'Is' for am/will with 1st person singular",
        "examples": [
            ("I am going to town.", "I's going to town.")
        ]
    },

    "would_distant_future": {
        "id": 119,
        "category": "modal_verbs",
        "description": "Would for (distant) future in contrast to will (immediate future)",
        "examples": [
            ("I will go eventually.", "I would go eventually.")
        ]
    },

    "would_if_clauses": {
        "id": 120,
        "category": "modal_verbs",
        "description": "Would in if-clauses",
        "examples": [
            ("If I were you...", "If I'd be you..."),
            ("If I had money, I would buy it.", "If I would have money, I would buy it.")
        ]
    },

    "double_modals": {
        "id": 121,
        "category": "modal_verbs",
        "description": "Double modals",
        "examples": [
            ("I tell you what we should do.", "I tell you what we might should do."),
            ("I might go.", "I might could go."),
            ("You should try.", "You might should try.")
        ]
    },

    "epistemic_mustnt": {
        "id": 122,
        "category": "modal_verbs",
        "description": "Epistemic mustn't (= can't be true)",
        "examples": [
            ("This can't be true.", "This mustn't be true.")
        ]
    },

    "present_modals_for_past": {
        "id": 123,
        "category": "modal_verbs",
        "description": "Present tense forms of modals where StE has past tense forms",
        "examples": [
            ("He might have gone.", "He might went.")
        ]
    },

    "want_need_participle": {
        "id": 124,
        "category": "modal_verbs",
        "description": "Want/need + past participle",
        "examples": [
            ("The cat wants to be petted.", "The cat wants petted."),
            ("The car needs to be washed.", "The car needs washed.")
        ]
    },

    "quasi_modals_core": {
        "id": 125,
        "category": "modal_verbs",
        "description": "New quasi-modals: core modal meanings (liketa = almost)",
        "examples": [
            ("We almost drowned that day.", "We liketa drowned that day.")
        ]
    },

    "quasi_modals_aspectual": {
        "id": 126,
        "category": "modal_verbs",
        "description": "New quasi-modals: aspectual meanings (fixin' to, finna)",
        "examples": [
            ("They're about to leave town.", "They're fixin' to leave town."),
            ("I am about to go.", "I'm finna go.")
        ]
    },

    "modals_politeness": {
        "id": 127,
        "category": "modal_verbs",
        "description": "Non-standard use of modals for politeness",
        "examples": [
            ("Shall I make you some tea?", "Must I make you some tea?")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 5: VERB MORPHOLOGY (eWAVE Features 128-153)
    # Past tense, participles, verb prefixes, serial verbs, copula forms
    # ═══════════════════════════════════════════════════════════════════════════

    "regularized_past": {
        "id": 128,
        "category": "verb_morphology",
        "description": "Levelling: regularization of irregular verb paradigms (-ed)",
        "examples": [
            ("I caught it.", "I catched it."),
            ("She knew the answer.", "She knowed the answer.")
        ]
    },

    "unmarked_past": {
        "id": 129,
        "category": "verb_morphology",
        "description": "Levelling: unmarked forms (base form for past)",
        "examples": [
            ("I gave it to her.", "I give it to her."),
            ("He ran away.", "He run away.")
        ]
    },

    "past_for_participle": {
        "id": 130,
        "category": "verb_morphology",
        "description": "Levelling: past tense replacing the past participle",
        "examples": [
            ("He had gone.", "He had went."),
            ("I have eaten.", "I have ate.")
        ]
    },

    "participle_for_past": {
        "id": 131,
        "category": "verb_morphology",
        "description": "Levelling: past participle replacing the past tense",
        "examples": [
            ("He went to Mary.", "He gone to Mary."),
            ("I saw it.", "I seen it.")
        ]
    },

    "zero_past_regular": {
        "id": 132,
        "category": "verb_morphology",
        "description": "Zero past tense forms of regular verbs",
        "examples": [
            ("I walked home.", "I walk home yesterday.")
        ]
    },

    "double_past": {
        "id": 133,
        "category": "verb_morphology",
        "description": "Double marking of past tense",
        "examples": [
            ("He came.", "He camed."),
            ("I didn't stay.", "I didn't stayed.")
        ]
    },

    "a_prefixing_ing": {
        "id": 134,
        "category": "verb_morphology",
        "description": "A-prefixing on -ing forms",
        "examples": [
            ("They weren't doing anything wrong.", "They wasn't a-doin' nothin' wrong."),
            ("He was running.", "He was a-running.")
        ]
    },

    "a_prefixing_other": {
        "id": 135,
        "category": "verb_morphology",
        "description": "A-prefixing on elements other than -ing forms",
        "examples": [
            ("He came back.", "He acome back."),
            ("Going back.", "A-back.")
        ]
    },

    "special_be_forms": {
        "id": 136,
        "category": "verb_morphology",
        "description": "Special inflected forms of be",
        "examples": [
            ("It be that way.", "It bees that way.")
        ]
    },

    "special_do_forms": {
        "id": 137,
        "category": "verb_morphology",
        "description": "Special inflected forms of do",
        "examples": [
            ("I don't.", "I junt.")
        ]
    },

    "special_have_forms": {
        "id": 138,
        "category": "verb_morphology",
        "description": "Special inflected forms of have",
        "examples": [
            ("I have it.", "I han it.")
        ]
    },

    "distinct_aux_vs_full": {
        "id": 139,
        "category": "verb_morphology",
        "description": "Distinctive forms for auxiliary vs. full verb meanings",
        "examples": [
            ("She did it.", "She done it."),
            ("We have a muck.", "We has a muck.")
        ]
    },

    "other_copula_np": {
        "id": 140,
        "category": "verb_morphology",
        "description": "Other forms/phrases for copula 'be': before NPs",
        "examples": [
            ("Every day is a fishing day.", "Ebry day da fishing day.")
        ]
    },

    "other_copula_locative": {
        "id": 141,
        "category": "verb_morphology",
        "description": "Other forms/phrases for copula 'be': before locatives",
        "examples": [
            ("The children were at school.", "Den pikin ben de na skoro.")
        ]
    },

    "other_copula_adjp": {
        "id": 142,
        "category": "verb_morphology",
        "description": "Other forms/phrases for copula 'be': before AdjPs",
        "examples": [
            ("She is sick.", "Shi stei sik.")
        ]
    },

    "transitive_suffix": {
        "id": 143,
        "category": "verb_morphology",
        "description": "Transitive verb suffix -em/-im/-um",
        "examples": [
            ("I bought some food.", "Mi bin bai-im kaikai.")
        ]
    },

    "gotten_got_distinction": {
        "id": 144,
        "category": "verb_morphology",
        "description": "Use of gotten and got with distinct meanings (dynamic vs. static)",
        "examples": [
            ("I have gotten a new job.", "I have gotten a new job."),
            ("I have got a car.", "I have got a car.")
        ]
    },

    "gotten_for_got": {
        "id": 145,
        "category": "verb_morphology",
        "description": "Use of gotten instead of got",
        "examples": [
            ("Finbank has got a new career website.", "Finbank has gotten a new career website.")
        ]
    },

    "ing_nonparticiple": {
        "id": 146,
        "category": "verb_morphology",
        "description": "Use of verbal suffix -ing with forms other than present participle/gerund",
        "examples": [
            ("I can drive now.", "I can driving now.")
        ]
    },

    "was_for_were": {
        "id": 147,
        "category": "verb_morphology",
        "description": "Was for conditional were",
        "examples": [
            ("If I were you.", "If I was you.")
        ]
    },

    "serial_verb_give": {
        "id": 148,
        "category": "verb_morphology",
        "description": "Serial verbs: give = 'to, for' (benefactive)",
        "examples": [
            ("Give the book to me.", "Karibuk giv mi."),
            ("Buy a book for me.", "Buy give me a book.")
        ]
    },

    "serial_verb_go": {
        "id": 149,
        "category": "verb_morphology",
        "description": "Serial verbs: go = 'movement away from'",
        "examples": [
            ("Are you taking the bus to Kingston?", "Yu a tek di bos go Kingstan?")
        ]
    },

    "serial_verb_come": {
        "id": 150,
        "category": "verb_morphology",
        "description": "Serial verbs: come = 'movement towards'",
        "examples": [
            ("They brought them back.", "Den bring den kam.")
        ]
    },

    "serial_verb_3": {
        "id": 151,
        "category": "verb_morphology",
        "description": "Serial verbs: constructions with 3 verbs",
        "examples": [
            ("He wants me to bring him.", "Im wan mi fi go kya im kom.")
        ]
    },

    "serial_verb_4plus": {
        "id": 152,
        "category": "verb_morphology",
        "description": "Serial verbs: constructions with 4 or more verbs",
        "examples": [
            ("Agnes rushed out to drop her mother off at the market.", "Agnes ron komot go lef in mama na makit.")
        ]
    },

    "give_passive": {
        "id": 153,
        "category": "verb_morphology",
        "description": "Give passive: NP1 (patient) + give + NP2 (agent) + V",
        "examples": [
            ("John was scolded by his boss.", "John give his boss scold.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 6: NEGATION (eWAVE Features 154-169)
    # Negative concord, ain't, invariant tags, special negators
    # ═══════════════════════════════════════════════════════════════════════════

    "negative_concord": {
        "id": 154,
        "category": "negation",
        "description": "Multiple negation / negative concord",
        "examples": [
            ("He won't do any harm.", "He won't do no harm."),
            ("I didn't see anyone.", "I didn't see nobody."),
            ("She never said anything to anyone.", "She never said nothing to nobody.")
        ]
    },

    "aint_be": {
        "id": 155,
        "category": "negation",
        "description": "Ain't as the negated form of be",
        "examples": [
            ("They're all in there, aren't they?", "They're all in there, ain't they?"),
            ("I am not going.", "I ain't going.")
        ]
    },

    "aint_have": {
        "id": 156,
        "category": "negation",
        "description": "Ain't as the negated form of have",
        "examples": [
            ("I haven't had a look at them yet.", "I ain't had a look at them yet.")
        ]
    },

    "aint_main_verb": {
        "id": 157,
        "category": "negation",
        "description": "Ain't as generic negator before a main verb",
        "examples": [
            ("Something I don't know about.", "Something I ain't know about.")
        ]
    },

    "invariant_dont": {
        "id": 158,
        "category": "negation",
        "description": "Invariant don't for all persons in present tense",
        "examples": [
            ("He doesn't like me.", "He don't like me."),
            ("She doesn't know.", "She don't know.")
        ]
    },

    "never_past_negator": {
        "id": 159,
        "category": "negation",
        "description": "Never as preverbal past tense negator (= didn't)",
        "examples": [
            ("He didn't come.", "He never came.")
        ]
    },

    "no_preverbal": {
        "id": 160,
        "category": "negation",
        "description": "No as preverbal negator",
        "examples": [
            ("I didn't eat breakfast.", "Me no iit brekfus.")
        ]
    },

    "not_preverbal": {
        "id": 161,
        "category": "negation",
        "description": "Not as a preverbal negator",
        "examples": [
            ("Nails don't float.", "Nail not float.")
        ]
    },

    "nomo_existential": {
        "id": 162,
        "category": "negation",
        "description": "No more/nomo as negative existential marker",
        "examples": [
            ("There isn't anything in there.", "Nomo nating insai dea.")
        ]
    },

    "was_werent_split": {
        "id": 163,
        "category": "negation",
        "description": "Was-weren't split (weren't for singular, wasn't for plural, or vice versa)",
        "examples": [
            ("The boys were interested, but Mary wasn't.", "The boys was interested, but Mary weren't.")
        ]
    },

    "amnt_tag": {
        "id": 164,
        "category": "negation",
        "description": "Amn't in tag questions",
        "examples": [
            ("I'm here, aren't I?", "I'm here, amn't I?")
        ]
    },

    "invariant_tag": {
        "id": 165,
        "category": "negation",
        "description": "Invariant non-concord tags (innit, isn't it)",
        "examples": [
            ("They had them in their hair, didn't they?", "They had them in their hair, innit?"),
            ("She is coming, isn't she?", "She is coming, isn't it?")
        ]
    },

    "can_or_not_tag": {
        "id": 166,
        "category": "negation",
        "description": "Invariant tag 'can or not?'",
        "examples": [
            ("Can I go home?", "I want to go home, can or not?")
        ]
    },

    "fronted_invariant_tag": {
        "id": 167,
        "category": "negation",
        "description": "Fronted invariant tag",
        "examples": [
            ("I can color this red, can't I?", "Isn't I can colour this red?")
        ]
    },

    "special_negative_imperatives": {
        "id": 168,
        "category": "negation",
        "description": "Special negative verbs in imperatives",
        "examples": [
            ("Don't make a face.", "Du miek agli.")
        ]
    },

    "negative_question_responses": {
        "id": 169,
        "category": "negation",
        "description": "Non-standard system underlying responses to negative yes/no questions",
        "examples": [
            ("Isn't he here? No (= he isn't).", "Isn't he here? Yes (= he isn't).")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 7: AGREEMENT (eWAVE Features 170-184)
    # Subject-verb agreement, copula deletion, existentials
    # ═══════════════════════════════════════════════════════════════════════════

    "zero_3sg": {
        "id": 170,
        "category": "agreement",
        "description": "Invariant present tense due to zero marking for third person singular",
        "examples": [
            ("So she shows up and says...", "So she show up and say..."),
            ("He walks to school.", "He walk to school.")
        ]
    },

    "generalized_3sg_s": {
        "id": 171,
        "category": "agreement",
        "description": "Invariant present tense due to generalization of 3rd person -s to all persons",
        "examples": [
            ("I see the house.", "I sees the house."),
            ("We go to school.", "We goes to school.")
        ]
    },

    "there_singular_plural": {
        "id": 172,
        "category": "agreement",
        "description": "Existential there's/there is/there was with plural subjects",
        "examples": [
            ("There are two men waiting in the hall.", "There's two men waiting in the hall.")
        ]
    },

    "variant_existential": {
        "id": 173,
        "category": "agreement",
        "description": "Variant forms of dummy subject there in existential clauses",
        "examples": [
            ("There is something wrong with her.", "They is something bad wrong with her."),
            ("There is a new person here.", "It's a new person here.")
        ]
    },

    "delete_aux_progressive": {
        "id": 174,
        "category": "agreement",
        "description": "Deletion of auxiliary be before progressive",
        "examples": [
            ("So you are always thinking about work.", "So you always thinking about where you go to work."),
            ("She is running.", "She running.")
        ]
    },

    "delete_aux_gonna": {
        "id": 175,
        "category": "agreement",
        "description": "Deletion of auxiliary be before gonna",
        "examples": [
            ("I am going to go work.", "I gonna go work.")
        ]
    },

    "delete_copula_np": {
        "id": 176,
        "category": "agreement",
        "description": "Deletion of copula be before NPs",
        "examples": [
            ("He is a good teacher.", "He a good teacher.")
        ]
    },

    "delete_copula_adjp": {
        "id": 177,
        "category": "agreement",
        "description": "Deletion of copula be before AdjPs",
        "examples": [
            ("She is smart.", "She smart."),
            ("He is tall.", "He tall.")
        ]
    },

    "delete_copula_locative": {
        "id": 178,
        "category": "agreement",
        "description": "Deletion of copula be before locatives",
        "examples": [
            ("She is at home.", "She at home.")
        ]
    },

    "delete_aux_have": {
        "id": 179,
        "category": "agreement",
        "description": "Deletion of auxiliary have",
        "examples": [
            ("I have eaten my lunch.", "I eaten my lunch.")
        ]
    },

    "was_were_generalization": {
        "id": 180,
        "category": "agreement",
        "description": "Was/were generalization",
        "examples": [
            ("You were hungry but he was thirsty.", "You was hungry but he was thirsty."),
            ("We were happy.", "We was happy.")
        ]
    },

    "agreement_subject_type": {
        "id": 181,
        "category": "agreement",
        "description": "Agreement sensitive to subject type",
        "examples": [
            ("Birds sing vs. they sing.", "Birds sings vs. they sing.")
        ]
    },

    "agreement_subject_position": {
        "id": 182,
        "category": "agreement",
        "description": "Agreement sensitive to position of subject",
        "examples": [
            ("I sing and dance.", "I sing and dances.")
        ]
    },

    "northern_subject_rule": {
        "id": 183,
        "category": "agreement",
        "description": "Northern Subject Rule",
        "examples": [
            ("I sing but birds sing.", "I sing but birds sings; I sing and dances.")
        ]
    },

    "invariant_be_nonhabitual": {
        "id": 184,
        "category": "agreement",
        "description": "Invariant be with non-habitual function",
        "examples": [
            ("Here I am.", "Here I be."),
            ("I am cold.", "I be cold.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 8: RELATIVIZATION (eWAVE Features 185-199)
    # Relative pronouns, zero relatives, resumptives
    # ═══════════════════════════════════════════════════════════════════════════

    "that_what_nonrestrictive": {
        "id": 185,
        "category": "relativization",
        "description": "Relativizer that or what in non-restrictive contexts",
        "examples": [
            ("My daughter, who lives in London...", "My daughter, that/what lives in London...")
        ]
    },

    "which_for_who": {
        "id": 186,
        "category": "relativization",
        "description": "Which for 'who' (human antecedents)",
        "examples": [
            ("My brother, who...", "My brother, which...")
        ]
    },

    "relativizer_as": {
        "id": 187,
        "category": "relativization",
        "description": "Relativizer as",
        "examples": [
            ("He was a chap who got a living anyhow.", "He was a chap as got a living anyhow.")
        ]
    },

    "relativizer_at": {
        "id": 188,
        "category": "relativization",
        "description": "Relativizer at",
        "examples": [
            ("This is the man who painted my house.", "This is the man at painted my house.")
        ]
    },

    "relativizer_where": {
        "id": 189,
        "category": "relativization",
        "description": "Relativizer where or form derived from where",
        "examples": [
            ("The Underground Railroad that helped the slaves.", "The Underground Railroad where help de slaves to run way.")
        ]
    },

    "relativizer_what": {
        "id": 190,
        "category": "relativization",
        "description": "Relativizer what or form derived from what",
        "examples": [
            ("This is the man who painted my house.", "This is the man what painted my house.")
        ]
    },

    "relativizer_doubling": {
        "id": 191,
        "category": "relativization",
        "description": "Relativizer doubling",
        "examples": [
            ("These little fellows that had stayed before God praying.", "These little fellahs that which had stayed befo' God prayin'.")
        ]
    },

    "analytic_whose": {
        "id": 192,
        "category": "relativization",
        "description": "Use of analytic that his/what his/at's instead of whose",
        "examples": [
            ("The man whose wife has died.", "The man what's wife has died.")
        ]
    },

    "zero_relative_subject": {
        "id": 193,
        "category": "relativization",
        "description": "Gapping/zero-relativization in subject position",
        "examples": [
            ("The man who lives there is a nice chap.", "The man lives there is a nice chap.")
        ]
    },

    "resumptive_pronouns": {
        "id": 194,
        "category": "relativization",
        "description": "Resumptive/shadow pronouns",
        "examples": [
            ("This is the house which I painted.", "This is the house which I painted it yesterday.")
        ]
    },

    "one_relativizer": {
        "id": 195,
        "category": "relativization",
        "description": "Postposed one as sole relativizer",
        "examples": [
            ("That boy who pinched my sister is very naughty.", "That boy pinch my sister one very naughty.")
        ]
    },

    "correlative_constructions": {
        "id": 196,
        "category": "relativization",
        "description": "Correlative constructions",
        "examples": [
            ("The ones I put in the jar are best.", "Which-one I put in the jar, that-one is good.")
        ]
    },

    "linking_relative": {
        "id": 197,
        "category": "relativization",
        "description": "Linking relative clauses",
        "examples": [
            ("Some universities are not going to give those marks.", "Unless you are going to get 88 which some universities are not going to give those marks.")
        ]
    },

    "preposition_chopping": {
        "id": 198,
        "category": "relativization",
        "description": "Deletion of stranded prepositions in relative clauses",
        "examples": [
            ("A big yard that you do gardening in.", "Like a big yard that you do gardening an'all.")
        ]
    },

    "reduced_relative_prenominal": {
        "id": 199,
        "category": "relativization",
        "description": "Reduced relative phrases preceding head-noun",
        "examples": [
            ("That jersey which Neela knitted is gone white.", "That Neela's-knitted jersey is gone white.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 9: COMPLEMENTATION (eWAVE Features 200-210)
    # Complementizers, infinitives
    # ═══════════════════════════════════════════════════════════════════════════

    "say_complementizer": {
        "id": 200,
        "category": "complementation",
        "description": "Say-based complementizers",
        "examples": [
            ("We heard that you went to the city.", "We hear say you gone to da city.")
        ]
    },

    "for_complementizer": {
        "id": 201,
        "category": "complementation",
        "description": "For-based complementizers",
        "examples": [
            ("It's hard to cross the river.", "I hard fi kraas di riba.")
        ]
    },

    "for_to_infinitive": {
        "id": 202,
        "category": "complementation",
        "description": "Unsplit for to in infinitival purpose clauses",
        "examples": [
            ("We always had gutters to drain the water away.", "We always had gutters in the winter time for to drain the water away.")
        ]
    },

    "for_infinitive_marker": {
        "id": 203,
        "category": "complementation",
        "description": "For (to) as infinitive marker",
        "examples": [
            ("You weren't allowed to take another job.", "You werenae allowed at this time for to go and take another job on.")
        ]
    },

    "what_than_comparative": {
        "id": 204,
        "category": "complementation",
        "description": "As what / than what in comparative clauses",
        "examples": [
            ("It's harder than you think.", "It's harder than what you think it is.")
        ]
    },

    "existential_get": {
        "id": 205,
        "category": "complementation",
        "description": "Existentials with forms of get",
        "examples": [
            ("There is some sand.", "E got some sand there.")
        ]
    },

    "existential_have": {
        "id": 206,
        "category": "complementation",
        "description": "Existentials with forms of have",
        "examples": [
            ("There are some women whose husbands have died.", "But there are some women whose husbands have already died.")
        ]
    },

    "that_for_infinitive": {
        "id": 207,
        "category": "complementation",
        "description": "Substitution of that-clause for infinitival subclause",
        "examples": [
            ("I wanted to get leave.", "I wanted that I should get leave.")
        ]
    },

    "deletion_to_infinitive": {
        "id": 208,
        "category": "complementation",
        "description": "Deletion of to before infinitives",
        "examples": [
            ("Allow him to go.", "Allow him go.")
        ]
    },

    "addition_to_infinitive": {
        "id": 209,
        "category": "complementation",
        "description": "Addition of to where StE has bare infinitive",
        "examples": [
            ("He made me do it.", "He made me to do it.")
        ]
    },

    "bare_root_complement": {
        "id": 210,
        "category": "complementation",
        "description": "Non-finite clause complements with bare root rather than -ing form",
        "examples": [
            ("He started telling the cousins.", "Him start tell di cousins.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 10: ADVERBIAL SUBORDINATION (eWAVE Features 211-215)
    # Conjunctions, clause-final particles
    # ═══════════════════════════════════════════════════════════════════════════

    "clause_final_but_though": {
        "id": 211,
        "category": "adverbial_subordination",
        "description": "Clause-final but = 'though'",
        "examples": [
            ("Well I wasn't so very cold though.", "Well I warnt so very cold but.")
        ]
    },

    "clause_final_but_really": {
        "id": 212,
        "category": "adverbial_subordination",
        "description": "Clause-final but = 'really'",
        "examples": [
            ("I'm really afraid of dogs!", "I fright for dogs, but eh!")
        ]
    },

    "no_subordination_chaining": {
        "id": 213,
        "category": "adverbial_subordination",
        "description": "No subordination; chaining construction linking two main verbs",
        "examples": [
            ("I went there to work.", "I bin go dere work.")
        ]
    },

    "conjunction_doubling": {
        "id": 214,
        "category": "adverbial_subordination",
        "description": "Conjunction doubling: clause + conj. + conj. + clause",
        "examples": [
            ("He has been in this school for five years, still he is not tired.", "He has been in this school for five years, still yet he is not tired.")
        ]
    },

    "correlative_conjunction_doubling": {
        "id": 215,
        "category": "adverbial_subordination",
        "description": "Conjunction doubling: correlative conjunctions",
        "examples": [
            ("Although you are smart, you are not appreciated.", "Although you are smart, but you are not appreciated.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 11: ADVERBS AND PREPOSITIONS (eWAVE Features 216-222)
    # Flat adverbs, preposition dropping
    # ═══════════════════════════════════════════════════════════════════════════

    "omit_prepositions": {
        "id": 216,
        "category": "adverbs_prepositions",
        "description": "Omission of StE prepositions",
        "examples": [
            ("He came out of hospital.", "He came out hospital."),
            ("She went to town.", "She went town.")
        ]
    },

    "postpositions": {
        "id": 217,
        "category": "adverbs_prepositions",
        "description": "Use of postpositions",
        "examples": [
            ("Under the chalkboard.", "The chalkboard under."),
            ("At night.", "Night time.")
        ]
    },

    "affirmative_anymore": {
        "id": 218,
        "category": "adverbs_prepositions",
        "description": "Affirmative anymore = 'nowadays'",
        "examples": [
            ("Nowadays they have a hard time protecting things.", "Anymore they have a hard time protecting things like that.")
        ]
    },

    "way_time_adverbs": {
        "id": 219,
        "category": "adverbs_prepositions",
        "description": "Adverb-forming suffixes -way and -time",
        "examples": [
            ("Far away.", "Long-way."),
            ("Quickly.", "Quick-way."),
            ("At night.", "Dark-time.")
        ]
    },

    "degree_adverb_adjective_form": {
        "id": 220,
        "category": "adverbs_prepositions",
        "description": "Degree modifier adverbs have the same form as adjectives",
        "examples": [
            ("That's really good.", "That's real good."),
            ("He is extremely tall.", "He is real tall.")
        ]
    },

    "flat_adverbs": {
        "id": 221,
        "category": "adverbs_prepositions",
        "description": "Other adverbs have the same form as adjectives",
        "examples": [
            ("Come quickly!", "Come quick!"),
            ("He runs fast.", "He runs quick.")
        ]
    },

    "too_very_intensifier": {
        "id": 222,
        "category": "adverbs_prepositions",
        "description": "Too; too much; very much = 'very' as qualifier",
        "examples": [
            ("It is very difficult.", "It is too difficult.")
        ]
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORY 12: DISCOURSE AND WORD ORDER (eWAVE Features 223-235)
    # Focus, topicalization, inversion, quotatives
    # ═══════════════════════════════════════════════════════════════════════════

    "other_clefting": {
        "id": 223,
        "category": "discourse_word_order",
        "description": "Other options for clefting than StE",
        "examples": [
            ("A lot of them are looking for more land.", "It's looking for more land a lot of them are.")
        ]
    },

    "other_fronting": {
        "id": 224,
        "category": "discourse_word_order",
        "description": "Other possibilities for fronting than StE",
        "examples": [
            ("Sometimes I speak English to my sister.", "To my sister sometime I speak English.")
        ]
    },

    "focus_marker": {
        "id": 225,
        "category": "discourse_word_order",
        "description": "Sentence-initial focus marker",
        "examples": [
            ("It is JOHN who did it.", "Na John do am.")
        ]
    },

    "negative_inversion": {
        "id": 226,
        "category": "discourse_word_order",
        "description": "Negative inversion",
        "examples": [
            ("Nobody showed up.", "Didn't nobody show up."),
            ("Nobody can do that.", "Can't nobody do that.")
        ]
    },

    "inverted_indirect_questions": {
        "id": 227,
        "category": "discourse_word_order",
        "description": "Inverted word order in indirect questions",
        "examples": [
            ("I'm wondering what you're going to do.", "I'm wondering what are you gonna do.")
        ]
    },

    "no_inversion_wh": {
        "id": 228,
        "category": "discourse_word_order",
        "description": "No inversion/no auxiliaries in wh-questions",
        "examples": [
            ("What are you doing?", "What you doing?"),
            ("What does he want?", "What he wants?")
        ]
    },

    "no_inversion_yn": {
        "id": 229,
        "category": "discourse_word_order",
        "description": "No inversion/no auxiliaries in main clause yes/no questions",
        "examples": [
            ("Do you get the point?", "You get the point?"),
            ("Did you like India?", "You liked India?")
        ]
    },

    "doubly_filled_comp": {
        "id": 230,
        "category": "discourse_word_order",
        "description": "Doubly filled COMP-position with wh-words",
        "examples": [
            ("Who ate what?", "What who has eaten?")
        ]
    },

    "most_before_head": {
        "id": 231,
        "category": "discourse_word_order",
        "description": "Superlative marker most occurring before head noun",
        "examples": [
            ("The thing I like most is apples.", "The most thing I like is apples.")
        ]
    },

    "double_object_order": {
        "id": 232,
        "category": "discourse_word_order",
        "description": "Either order of objects in double object constructions (if both pronominal)",
        "examples": [
            ("He couldn't give it to him.", "He couldn't give him it."),
            ("I gave it back to her.", "I tan it her back.")
        ]
    },

    "subject_imperatives": {
        "id": 233,
        "category": "discourse_word_order",
        "description": "Presence of subject in imperatives",
        "examples": [
            ("Go there!", "Go you there!")
        ]
    },

    "like_focus": {
        "id": 234,
        "category": "discourse_word_order",
        "description": "Like as a focusing device",
        "examples": [
            ("For one, I found five pounds, that was like three pounds.", "Like for one found five quid, that was like three quid.")
        ]
    },

    "like_quotative": {
        "id": 235,
        "category": "discourse_word_order",
        "description": "Like as a quotative particle",
        "examples": [
            ("And she said 'What do you mean?'", "And she was like 'What do you mean?'")
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY DEFINITIONS (eWAVE 12 Categories)
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "pronouns": {
        "name": "Pronouns",
        "description": "Pronoun exchange, nominal gender, case marking, reflexives, possessives, address forms",
        "feature_range": "1-47"
    },
    "noun_phrase": {
        "name": "Noun Phrase",
        "description": "Plurals, articles, determiners, possession, comparison",
        "feature_range": "48-87"
    },
    "tense_aspect": {
        "name": "Tense & Aspect",
        "description": "Progressive, habitual, perfect, completive, future, tense markers",
        "feature_range": "88-113"
    },
    "modal_verbs": {
        "name": "Modal Verbs",
        "description": "Double modals, quasi-modals, future markers",
        "feature_range": "114-127"
    },
    "verb_morphology": {
        "name": "Verb Morphology",
        "description": "Past tense, participles, verb prefixes, serial verbs, copula forms",
        "feature_range": "128-153"
    },
    "negation": {
        "name": "Negation",
        "description": "Negative concord, ain't, invariant tags, special negators",
        "feature_range": "154-169"
    },
    "agreement": {
        "name": "Agreement",
        "description": "Subject-verb agreement, copula/auxiliary deletion, existentials",
        "feature_range": "170-184"
    },
    "relativization": {
        "name": "Relativization",
        "description": "Relative pronouns, zero relatives, resumptive pronouns",
        "feature_range": "185-199"
    },
    "complementation": {
        "name": "Complementation",
        "description": "Complementizers, infinitives",
        "feature_range": "200-210"
    },
    "adverbial_subordination": {
        "name": "Adverbial Subordination",
        "description": "Conjunctions, clause-final particles",
        "feature_range": "211-215"
    },
    "adverbs_prepositions": {
        "name": "Adverbs & Prepositions",
        "description": "Flat adverbs, preposition dropping, postpositions",
        "feature_range": "216-222"
    },
    "discourse_word_order": {
        "name": "Discourse Organization & Word Order",
        "description": "Focus, topicalization, inversion, quotatives",
        "feature_range": "223-235"
    }
}


def get_features_by_category(category: str) -> dict:
    """Get all features belonging to a specific category."""
    return {
        key: feat for key, feat in FEATURE_LIBRARY.items()
        if feat.get("category") == category
    }


def get_feature_by_id(feature_id: int) -> dict:
    """Get a feature by its eWAVE ID."""
    for key, feat in FEATURE_LIBRARY.items():
        if feat.get("id") == feature_id:
            return {key: feat}
    return None


def list_all_features() -> list:
    """List all feature keys."""
    return list(FEATURE_LIBRARY.keys())


def get_feature_count() -> int:
    """Get total number of features."""
    return len(FEATURE_LIBRARY)
