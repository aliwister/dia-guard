# dialects.py - Complete dialect definitions with feature mappings for 76 English dialects
# Aligned with eWAVE (Electronic World Atlas of Varieties of English) - https://ewave-atlas.org
# Feature keys reference features.py which contains all 235 eWAVE features

DIALECT_REGISTRY = {
    # ═══════════════════════════════════════════════════════════════
    # NORTH AMERICA
    # ═══════════════════════════════════════════════════════════════

    "urban_aave": {
        "name": "Urban African American Vernacular English",
        "region": "United States (Urban Areas)",
        "description": "Contemporary AAVE spoken in urban centers with extensive eWAVE documentation",
        "features": [
            # Pronouns (eWAVE Category 1)
            "me_coordinate_subjects",         # F7: Me and him went to the store
            "regularized_reflexives",         # F11: hisself, theirselves
            "second_person_plural",           # F34: y'all, you all
            "generalized_3sg_object",         # F6: Give it to she

            # Noun Phrase (eWAVE Category 2)
            "double_determiners",             # F59: them books, this here car
            "associative_plural_them",        # F52: John and them (= John's group)
            "zero_plural_nonhuman",           # F58: two mile, several year
            "zero_genitive",                  # F77: John car (possession by juxtaposition)
            "them_for_those",                 # F68: them books for those books

            # Tense/Aspect (eWAVE Category 3)
            "invariant_be_habitual",          # F90: He be working (habitual action)
            "completive_done",                # F104: I done finished
            "have_done_completive",           # F105: He is done gone
            "be_done_irrealis",               # F106: She be done had her baby (resultative)
            "been_past_anterior",             # F111: I been knowing him (remote past)

            # Modal Verbs (eWAVE Category 4)
            "go_future",                      # F114: I'm gonna/gon go
            "quasi_modals_aspectual",         # F126: finna, steady, come (aspectual)

            # Verb Morphology (eWAVE Category 5)
            "unmarked_past",                  # F129: He walk there yesterday
            "regularized_past",               # F128: knowed, growed
            "participle_for_past",            # F131: I seen him yesterday

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: I don't have no money
            "aint_be",                        # F155: ain't for isn't/aren't
            "aint_have",                      # F156: ain't for haven't/hasn't
            "invariant_dont",                 # F158: he don't know
            "never_past_negator",             # F159: I never went there yesterday
            "invariant_tag",                  # F165: isn't it, innit

            # Agreement (eWAVE Category 7)
            "zero_3sg",                       # F170: She walk to school
            "there_singular_plural",          # F172: There's two cars outside
            "variant_existential",            # F173: It's a man outside
            "delete_aux_progressive",         # F174: She Ø running
            "delete_aux_gonna",               # F175: I Ø gonna go
            "delete_copula_np",               # F176: He Ø a teacher
            "delete_copula_adjp",             # F177: She Ø smart
            "delete_copula_locative",         # F178: They Ø at home
            "was_were_generalization",        # F180: We was happy

            # Relativization (eWAVE Category 8)
            "zero_relative_subject",          # F193: The man Ø came here
            "resumptive_pronouns",            # F194: The man that I saw him

            # Adverbs & Prepositions (eWAVE Category 11)
            "flat_adverbs",                   # F221: He run real quick

            # Discourse & Word Order (eWAVE Category 12)
            "negative_inversion",             # F226: Can't nobody do that
            "other_fronting",                 # F224: That book, I read it
            "like_quotative"                  # F235: He was like "what?"
        ],
        "strength": 1.0,
        "notes": "Most documented urban variety with rich aspectual system. Key markers: habitual be, completive done, copula deletion, negative concord, ain't."
    },

    "rural_aave": {
        "name": "Rural African American Vernacular English",
        "region": "United States (Rural South)",
        "description": "AAVE spoken in rural Southern communities with Southern dialect influence",
        "features": [
            # Pronouns (eWAVE Category 1)
            "me_coordinate_subjects",         # F7: Me and him went
            "regularized_reflexives",         # F11: hisself, theirselves
            "second_person_plural",           # F34: y'all
            "nasal_possessive",               # F25: hisn, hern, ourn

            # Noun Phrase (eWAVE Category 2)
            "associative_plural_them",        # F52: John and them
            "zero_plural_nonhuman",           # F58: two mile
            "them_for_those",                 # F68: them books

            # Tense/Aspect (eWAVE Category 3)
            "invariant_be_habitual",          # F90: He be working
            "completive_done",                # F104: I done finished
            "been_past_anterior",             # F111: I been knowing

            # Modal Verbs (eWAVE Category 4)
            "double_modals",                  # F123: might could, might would
            "quasi_modals_core",              # F125: liketa, fixin' to

            # Verb Morphology (eWAVE Category 5)
            "a_prefixing_ing",                # F134: a-going, a-hunting
            "regularized_past",               # F128: knowed, growed
            "participle_for_past",            # F131: I seen him

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: don't have nothing
            "aint_be",                        # F155: ain't for isn't
            "aint_have",                      # F156: ain't for haven't
            "invariant_dont",                 # F158: he don't know
            "never_past_negator",             # F159: never as past negator

            # Agreement (eWAVE Category 7)
            "zero_3sg",                       # F170: She walk to school
            "delete_copula_adjp",             # F177: She Ø smart
            "delete_copula_np",               # F176: He Ø a teacher
            "was_were_generalization",        # F180: We was happy
            "there_singular_plural"           # F172: There's two cars
        ],
        "strength": 0.8,
        "notes": "Shares features with both Urban AAVE and Southern dialects. A-prefixing and double modals from Appalachian contact."
    },

    "earlier_aave": {
        "name": "Earlier African American Vernacular English",
        "region": "United States (Historical)",
        "description": "Historical AAVE features from earlier periods",
        "features": [
            "delete_copula_adjp", "delete_copula_np", "completive_done",
            "negative_concord", "aint_be", "zero_3sg", "unmarked_past",
            "serial_verb_go", "variant_existential", "zero_genitive"
        ],
        "strength": 0.7,
        "notes": "Creole-influenced features more prominent"
    },

    "appalachian": {
        "name": "Appalachian English",
        "region": "United States (Appalachian Mountains)",
        "description": "Dialect of the Appalachian mountain region with archaic Scots-Irish features",
        "features": [
            # Pronouns (eWAVE Category 1)
            "me_coordinate_subjects",         # F7: Me and him went
            "regularized_reflexives",         # F11: hisself, theirselves
            "second_person_plural",           # F34: you'uns, you all
            "nasal_possessive",               # F25: hisn, hern, ourn, yourn
            "us_singular_object",             # F28: Give us a cookie

            # Noun Phrase (eWAVE Category 2)
            "double_determiners",             # F59: this here, that there
            "them_for_those",                 # F68: them books
            "double_comparatives",            # F79: more bigger, most bestest
            "zero_plural_nonhuman",           # F58: two mile

            # Tense/Aspect (eWAVE Category 3)
            "completive_done",                # F104: I done finished
            "been_past_anterior",             # F111: I been knowing

            # Modal Verbs (eWAVE Category 4)
            "double_modals",                  # F123: might could, might would, may can
            "quasi_modals_core",              # F125: liketa, fixin' to

            # Verb Morphology (eWAVE Category 5)
            "a_prefixing_ing",                # F134: a-going, a-hunting (iconic feature)
            "regularized_past",               # F128: knowed, growed, throwed
            "participle_for_past",            # F131: I seen him, I done it

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: I ain't got nothing
            "aint_be",                        # F155: ain't for isn't/aren't
            "aint_have",                      # F156: ain't for haven't
            "never_past_negator",             # F159: I never went (= didn't go)

            # Agreement (eWAVE Category 7)
            "there_singular_plural",          # F172: There's two boys
            "was_were_generalization",        # F180: We was, you was
            "zero_3sg",                       # F170: He walk (less common)

            # Complementation (eWAVE Category 9)
            "for_to_infinitive",              # F204: I want for to go

            # Adverbs & Prepositions (eWAVE Category 11)
            "flat_adverbs"                    # F221: He run real quick
        ],
        "strength": 1.0,
        "notes": "A-prefixing and double modals are iconic features. Scots-Irish heritage preserved in isolated mountain communities."
    },

    "ozark": {
        "name": "Ozark English",
        "region": "United States (Ozark Mountains)",
        "description": "Dialect of the Ozark mountain region",
        "features": [
            "a_prefixing_ing", "double_modals", "second_person_plural",
            "nasal_possessive", "negative_concord", "was_were_generalization",
            "double_comparatives", "regularized_reflexives", "aint_be",
            "for_to_infinitive"
        ],
        "strength": 0.9,
        "notes": "Similar to Appalachian with some distinct features"
    },

    "southeast_enclave": {
        "name": "Southeast American Enclave Dialects",
        "region": "United States (Southeast Islands/Isolated Areas)",
        "description": "Isolated dialect communities in the Southeast",
        "features": [
            "regularized_reflexives", "nasal_possessive", "for_to_infinitive",
            "double_modals", "a_prefixing_ing", "was_were_generalization",
            "negative_concord"
        ],
        "strength": 0.8,
        "notes": "Preserved archaic features"
    },

    "chicano": {
        "name": "Chicano English",
        "region": "United States (Southwest)",
        "description": "English variety of Mexican-American communities",
        "features": [
            "progressive_stative", "zero_for_indefinite", "zero_relative_subject",
            "negative_concord", "delete_aux_have", "regularized_past"
        ],
        "strength": 0.6,
        "notes": "Spanish-influenced phonology and some grammar"
    },

    "gullah": {
        "name": "Gullah (Sea Island Creole)",
        "region": "United States (Sea Islands, SC/GA)",
        "description": "English-based creole of the Sea Islands",
        "features": [
            "delete_copula_adjp", "delete_copula_np", "zero_3sg",
            "unmarked_past", "serial_verb_go", "serial_verb_give",
            "variant_existential", "zero_genitive", "zero_plural_nonhuman",
            "been_past_anterior", "completive_done", "negative_concord"
        ],
        "strength": 1.0,
        "notes": "Creole with strong West African influences"
    },

    "newfoundland": {
        "name": "Newfoundland English",
        "region": "Canada (Newfoundland)",
        "description": "Distinctive dialect of Newfoundland",
        "features": [
            "be_perfect_auxiliary", "after_perfect", "invariant_tag",
            "regularized_reflexives", "them_for_those", "second_person_plural",
            "invariant_be_habitual", "for_to_infinitive"
        ],
        "strength": 0.8,
        "notes": "Irish English influence prominent"
    },

    "colloquial_american": {
        "name": "Colloquial American English",
        "region": "United States (General)",
        "description": "Informal American English features",
        "features": [
            "me_coordinate_subjects", "flat_adverbs", "invariant_dont",
            "negative_concord", "like_quotative", "there_singular_plural"
        ],
        "strength": 0.4,
        "notes": "Common informal features across regions"
    },

    # ═══════════════════════════════════════════════════════════════
    # BRITISH ISLES
    # ═══════════════════════════════════════════════════════════════

    "scottish": {
        "name": "Scottish English",
        "region": "Scotland",
        "description": "English as spoken in Scotland with Scots language influence",
        "features": [
            # Pronouns (eWAVE Category 1)
            "second_person_plural",           # F34: youse, ye
            "regularized_reflexives",         # F11: hissel, theirsel

            # Tense/Aspect (eWAVE Category 3)
            "progressive_stative",            # F88: I'm wanting, I'm knowing

            # Modal Verbs (eWAVE Category 4)
            "double_modals",                  # F123: might could, will can

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: I don't have nothing
            "never_past_negator",             # F159: I never saw him yesterday
            "no_preverbal",                   # F160: I'm no going
            "invariant_tag",                  # F165: is it no?

            # Agreement (eWAVE Category 7)
            "was_were_generalization",        # F180: We was, you was

            # Complementation (eWAVE Category 9)
            "for_to_infinitive",              # F204: for to go

            # Discourse & Word Order (eWAVE Category 12)
            "other_fronting"                  # F224: Topicalization
        ],
        "strength": 0.7,
        "notes": "Scots language influence. Key markers: no as negator, double modals, youse plural."
    },

    "irish": {
        "name": "Irish English (Hiberno-English)",
        "region": "Ireland",
        "description": "English as spoken in Ireland with strong Irish Gaelic substrate",
        "features": [
            # Pronouns (eWAVE Category 1)
            "second_person_plural",           # F34: ye, youse
            "regularized_reflexives",         # F11: hissel, theirself

            # Tense/Aspect (eWAVE Category 3)
            "be_perfect_auxiliary",           # F99: I'm finished (resultative)
            "after_perfect",                  # F102: I'm after eating (= I just ate)
            "progressive_stative",            # F88: I'm knowing, I'm wanting
            "do_tense_marker",                # F103: I do be working (habitual)

            # Negation (eWAVE Category 6)
            "invariant_tag",                  # F165: so it is, so he did
            "amnt_tag",                       # F166: amn't I?
            "negative_concord",               # F154: I didn't see nothing

            # Complementation (eWAVE Category 9)
            "for_to_infinitive",              # F204: for to go

            # Adverbial Subordination (eWAVE Category 10)
            "conjunction_doubling",           # F213: the reason is because

            # Discourse & Word Order (eWAVE Category 12)
            "other_clefting",                 # F223: It's himself that did it
            "other_fronting",                 # F224: A great man he was
            "focus_marker"                    # F225: Cleft focus constructions
        ],
        "strength": 0.8,
        "notes": "Strong Irish Gaelic substrate. Key markers: after perfect (I'm after V-ing), habitual do be, amn't, clefting."
    },

    "welsh": {
        "name": "Welsh English",
        "region": "Wales",
        "description": "English as spoken in Wales with Welsh language substrate",
        "features": [
            # Tense/Aspect (eWAVE Category 3)
            "progressive_stative",            # F88: I'm liking that
            "do_tense_marker",                # F103: I do like it (emphatic)

            # Negation (eWAVE Category 6)
            "invariant_tag",                  # F165: isn't it?
            "negative_concord",               # F154: I didn't see nothing

            # Discourse & Word Order (eWAVE Category 12)
            "other_fronting",                 # F224: A good man he is
            "other_clefting",                 # F223: It's the book that I want
            "focus_marker",                   # F225: Focus fronting

            # Adverbs & Prepositions (eWAVE Category 11)
            "clause_final_but_though"         # F212: Nice day, but
        ],
        "strength": 0.6,
        "notes": "Welsh language influence on word order (VSO substrate). Focus fronting and emphatic do common."
    },

    "northern_england": {
        "name": "Northern English Dialects",
        "region": "England (North)",
        "description": "Dialects of Northern England",
        "features": [
            "was_were_generalization", "negative_concord", "invariant_tag",
            "them_for_those", "regularized_reflexives", "us_singular_object",
            "double_object_order"
        ],
        "strength": 0.7,
        "notes": "Distinct from Southern British English"
    },

    "southwest_england": {
        "name": "Southwest English Dialects",
        "region": "England (Southwest)",
        "description": "Dialects of Southwest England",
        "features": [
            "be_perfect_auxiliary", "regularized_reflexives", "she_inanimate",
            "progressive_stative", "do_tense_marker"
        ],
        "strength": 0.6,
        "notes": "Archaic features preserved"
    },

    "east_anglian": {
        "name": "East Anglian English",
        "region": "England (East Anglia)",
        "description": "Dialect of East Anglia",
        "features": [
            "do_tense_marker", "regularized_reflexives",
            "negative_concord", "zero_3sg", "them_for_those"
        ],
        "strength": 0.5,
        "notes": "Distinctive 'do' usage"
    },

    "southeast_england": {
        "name": "Southeast English Dialects",
        "region": "England (Southeast)",
        "description": "Dialects of Southeast England (including Cockney features)",
        "features": [
            "them_for_those", "negative_concord", "invariant_tag",
            "me_coordinate_subjects", "aint_be"
        ],
        "strength": 0.5,
        "notes": "London influence"
    },

    "orkney_shetland": {
        "name": "Orkney and Shetland English",
        "region": "Scotland (Northern Isles)",
        "description": "English of Orkney and Shetland Islands",
        "features": [
            "regularized_reflexives", "be_perfect_auxiliary", "invariant_tag",
            "second_person_singular"
        ],
        "strength": 0.6,
        "notes": "Norse substrate influence"
    },

    "manx": {
        "name": "Manx English",
        "region": "Isle of Man",
        "description": "English of the Isle of Man",
        "features": [
            "be_perfect_auxiliary", "after_perfect", "progressive_stative",
            "invariant_tag", "relativizer_at"
        ],
        "strength": 0.5,
        "notes": "Celtic substrate"
    },

    "channel_islands": {
        "name": "Channel Islands English",
        "region": "Channel Islands",
        "description": "English of Jersey and Guernsey",
        "features": [
            "regularized_reflexives", "invariant_tag"
        ],
        "strength": 0.3,
        "notes": "Norman French influence"
    },

    "british_creole": {
        "name": "British Creole",
        "region": "United Kingdom (Urban)",
        "description": "Caribbean-influenced urban British variety",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "negative_concord", "variant_existential", "zero_plural_nonhuman"
        ],
        "strength": 0.7,
        "notes": "Caribbean creole influence"
    },

    "maltese_english": {
        "name": "Maltese English",
        "region": "Malta",
        "description": "English as spoken in Malta",
        "features": [
            "progressive_stative", "invariant_tag", "zero_for_definite"
        ],
        "strength": 0.4,
        "notes": "Maltese and Italian influence"
    },

    # ═══════════════════════════════════════════════════════════════
    # CARIBBEAN
    # ═══════════════════════════════════════════════════════════════

    "jamaican": {
        "name": "Jamaican English/Creole",
        "region": "Jamaica",
        "description": "English and Creole continuum from basilect (Patwa) to acrolect with preverbal TMA markers",
        "features": [
            # Pronouns (eWAVE Category 1)
            "generalized_3sg_subject",        # F5: Im/em for he/she
            "generalized_3sg_object",         # F6: Give it to im
            "no_gender_distinction",          # F10: Im for he/she (no gender)
            "object_pronoun_subject",         # F30: Him go there

            # Noun Phrase (eWAVE Category 2)
            "zero_plural_nonhuman",           # F58: Two book, tree year
            "zero_genitive",                  # F77: John book (possession)
            "fi_possessive",                  # F24: fi mi (= my)
            "zero_for_definite",              # F62: Go a shop
            "zero_for_indefinite",            # F63: Im a teacher

            # Tense/Aspect (eWAVE Category 3)
            "unmarked_past",                  # F129: Mi go yesterday
            "been_past_anterior",             # F111: Mi did know (anterior)
            "stap_stay_progressive",          # F93: Im a/de run (progressive)

            # Verb Morphology (eWAVE Category 5)
            "zero_3sg",                       # F170: She walk, he talk
            "serial_verb_go",                 # F149: Go buy it
            "serial_verb_give",               # F148: Carry give im
            "serial_verb_come",               # F150: Come look

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: No get nothing
            "no_preverbal",                   # F160: Mi no go (preverbal no)

            # Agreement (eWAVE Category 7)
            "variant_existential",            # F173: Got/have existential
            "delete_copula_adjp",             # F177: She tall (no copula)
            "delete_copula_np",               # F176: Him a teacher
            "delete_copula_locative",         # F178: Dem deh a yaad

            # Relativization (eWAVE Category 8)
            "zero_relative_subject",          # F193: Di man Ø come
            "relativizer_what",               # F191: The man what came

            # Complementation (eWAVE Category 9)
            "say_complementizer",             # F200: Im seh im a come
            "deletion_to_infinitive",         # F209: Mi want Ø go
            "for_complementizer",             # F201: Fi go (purpose)

            # Discourse & Word Order (eWAVE Category 12)
            "focus_marker",                   # F225: A im do it (focus)
            "other_fronting"                  # F224: Topicalization
        ],
        "strength": 1.0,
        "notes": "Creole continuum from basilect to acrolect. Key markers: preverbal TMA (did/a/go), no copula, serial verbs, im/dem pronouns."
    },

    "bahamian": {
        "name": "Bahamian English/Creole",
        "region": "Bahamas",
        "description": "English and Creole of the Bahamas",
        "features": [
            "delete_copula_adjp", "zero_3sg", "negative_concord",
            "unmarked_past", "variant_existential", "zero_plural_nonhuman",
            "completive_done"
        ],
        "strength": 0.9,
        "notes": "Mix of AAVE and Caribbean creole features"
    },

    "trinidadian": {
        "name": "Trinidadian Creole",
        "region": "Trinidad and Tobago",
        "description": "Creole of Trinidad and Tobago",
        "features": [
            "delete_copula_adjp", "zero_3sg", "completive_done",
            "negative_concord", "serial_verb_go", "zero_plural_nonhuman",
            "do_habitual", "unmarked_past"
        ],
        "strength": 0.9,
        "notes": "Distinctive 'does' habitual marker"
    },

    "guyanese": {
        "name": "Guyanese Creole (Creolese)",
        "region": "Guyana",
        "description": "English-based creole of Guyana",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "negative_concord", "serial_verb_go", "generalized_3sg_object",
            "zero_plural_nonhuman", "variant_existential", "been_past_anterior"
        ],
        "strength": 0.9,
        "notes": "Deep creole features"
    },

    "barbadian": {
        "name": "Barbadian Creole (Bajan)",
        "region": "Barbados",
        "description": "Creole of Barbados",
        "features": [
            "delete_copula_adjp", "zero_3sg", "negative_concord",
            "unmarked_past", "do_habitual", "zero_plural_nonhuman"
        ],
        "strength": 0.8,
        "notes": "Less basilectal than other Caribbean creoles"
    },

    "belizean": {
        "name": "Belizean Creole",
        "region": "Belize",
        "description": "English-based creole of Belize",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "negative_concord", "serial_verb_go", "zero_plural_nonhuman",
            "variant_existential"
        ],
        "strength": 0.9,
        "notes": "Central American creole"
    },

    "vincentian": {
        "name": "Vincentian Creole",
        "region": "St. Vincent and the Grenadines",
        "description": "Creole of St. Vincent",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "negative_concord", "zero_plural_nonhuman", "variant_existential"
        ],
        "strength": 0.8,
        "notes": "Eastern Caribbean creole"
    },

    # ═══════════════════════════════════════════════════════════════
    # AFRICA
    # ═══════════════════════════════════════════════════════════════

    "nigerian": {
        "name": "Nigerian English",
        "region": "Nigeria",
        "description": "English as spoken in Nigeria with West African substrate influence",
        "features": [
            # Pronouns (eWAVE Category 1)
            "resumptive_pronouns",            # F194: The man that I saw him
            "second_person_plural",           # F34: You people, una

            # Noun Phrase (eWAVE Category 2)
            "zero_for_definite",              # F62: Go to market
            "zero_for_indefinite",            # F63: He is doctor
            "zero_plural_nonhuman",           # F58: Many book

            # Tense/Aspect (eWAVE Category 3)
            "progressive_stative",            # F88: I am knowing, I am having
            "simple_present_for_perfect",     # F101: I live here since 1990

            # Verb Morphology (eWAVE Category 5)
            "zero_3sg",                       # F170: He go, she come
            "serial_verb_go",                 # F149: Go bring it

            # Negation (eWAVE Category 6)
            "invariant_tag",                  # F165: isn't it? (invariant)

            # Relativization (eWAVE Category 8)
            "zero_relative_subject",          # F193: The man Ø came

            # Complementation (eWAVE Category 9)
            "say_complementizer",             # F200: He said that...
            "deletion_to_infinitive",         # F209: Want Ø go

            # Discourse & Word Order (eWAVE Category 12)
            "other_fronting",                 # F224: That one, I like it
            "inverted_indirect_questions",    # F227: Tell me what is it
            "focus_marker"                    # F225: Na him do it
        ],
        "strength": 0.8,
        "notes": "West African substrate influence. Features shared with Ghanaian and Cameroon English."
    },

    "nigerian_pidgin": {
        "name": "Nigerian Pidgin",
        "region": "Nigeria",
        "description": "Nigerian Pidgin English (Naija) - major West African creole",
        "features": [
            # Pronouns (eWAVE Category 1)
            "generalized_3sg_subject",        # F5: Im/e for he/she
            "generalized_3sg_object",         # F6: Give am
            "no_gender_distinction",          # F10: Im for he/she

            # Noun Phrase (eWAVE Category 2)
            "zero_plural_nonhuman",           # F58: Plenty book
            "zero_for_definite",              # F62: Go market
            "zero_for_indefinite",            # F63: Na teacher be dat

            # Tense/Aspect (eWAVE Category 3)
            "unmarked_past",                  # F129: I go yesterday
            "completive_done",                # F104: I don finish
            "been_past_anterior",             # F111: I bin sabi

            # Verb Morphology (eWAVE Category 5)
            "zero_3sg",                       # F170: E dey go
            "serial_verb_go",                 # F149: Go carry am
            "serial_verb_give",               # F148: Carry give am
            "serial_verb_come",               # F150: Come look

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: No get nothing
            "no_preverbal",                   # F160: I no go

            # Agreement (eWAVE Category 7)
            "variant_existential",            # F173: E get/dey
            "delete_copula_adjp",             # F177: E fine (no copula)
            "delete_copula_np",               # F176: Na teacher

            # Complementation (eWAVE Category 9)
            "say_complementizer",             # F200: E talk say...

            # Discourse & Word Order (eWAVE Category 12)
            "focus_marker",                   # F225: Na im do am (focus na)
            "other_fronting"                  # F224: Dat one, I like am
        ],
        "strength": 1.0,
        "notes": "Major West African creole spoken by 100+ million. Key markers: preverbal dey (progressive), don (completive), bin (past), na (focus)."
    },

    "ghanaian": {
        "name": "Ghanaian English",
        "region": "Ghana",
        "description": "English as spoken in Ghana",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "serial_verb_go", "resumptive_pronouns", "invariant_tag",
            "zero_plural_nonhuman", "say_complementizer"
        ],
        "strength": 0.7,
        "notes": "West African English variety"
    },

    "cameroon": {
        "name": "Cameroon English",
        "region": "Cameroon",
        "description": "English as spoken in Cameroon",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "serial_verb_go", "invariant_tag", "resumptive_pronouns"
        ],
        "strength": 0.6,
        "notes": "Francophone and Anglophone varieties"
    },

    "kenyan": {
        "name": "Kenyan English",
        "region": "Kenya",
        "description": "English as spoken in Kenya",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "invariant_tag", "resumptive_pronouns", "inverted_indirect_questions"
        ],
        "strength": 0.6,
        "notes": "East African English"
    },

    "tanzanian": {
        "name": "Tanzanian English",
        "region": "Tanzania",
        "description": "English as spoken in Tanzania",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "invariant_tag", "resumptive_pronouns"
        ],
        "strength": 0.5,
        "notes": "Swahili influence"
    },

    "ugandan": {
        "name": "Ugandan English",
        "region": "Uganda",
        "description": "English as spoken in Uganda",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "invariant_tag", "resumptive_pronouns"
        ],
        "strength": 0.5,
        "notes": "East African variety"
    },

    "liberian_settler": {
        "name": "Liberian Settler English",
        "region": "Liberia",
        "description": "English of Americo-Liberian settlers",
        "features": [
            "completive_done", "negative_concord", "aint_be",
            "zero_3sg", "second_person_plural", "delete_aux_have"
        ],
        "strength": 0.7,
        "notes": "AAVE-influenced variety"
    },

    "south_african_black": {
        "name": "Black South African English",
        "region": "South Africa",
        "description": "English of Black South Africans",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "resumptive_pronouns", "invariant_tag", "zero_plural_nonhuman",
            "other_fronting"
        ],
        "strength": 0.7,
        "notes": "Bantu language substrate"
    },

    "south_african_indian": {
        "name": "Indian South African English",
        "region": "South Africa",
        "description": "English of Indian South Africans",
        "features": [
            "progressive_stative", "invariant_tag", "resumptive_pronouns",
            "inverted_indirect_questions"
        ],
        "strength": 0.5,
        "notes": "Indian English influence"
    },

    "south_african_white": {
        "name": "White South African English",
        "region": "South Africa",
        "description": "English of White South Africans",
        "features": [
            "she_inanimate", "invariant_tag", "progressive_stative"
        ],
        "strength": 0.3,
        "notes": "Afrikaans influence"
    },

    "zimbabwean_white": {
        "name": "White Zimbabwean English",
        "region": "Zimbabwe",
        "description": "English of White Zimbabweans",
        "features": [
            "she_inanimate", "invariant_tag"
        ],
        "strength": 0.3,
        "notes": "Similar to SA White English"
    },

    # ═══════════════════════════════════════════════════════════════
    # SOUTH AND SOUTHEAST ASIA
    # ═══════════════════════════════════════════════════════════════

    "indian": {
        "name": "Indian English",
        "region": "India",
        "description": "English as spoken in India with Indo-Aryan/Dravidian substrate influence",
        "features": [
            # Pronouns (eWAVE Category 1)
            "resumptive_pronouns",            # F194: The man who I saw him
            "emphatic_reflexives_own",        # F22: My own self, his own self

            # Noun Phrase (eWAVE Category 2)
            "zero_for_definite",              # F62: I went to hospital
            "zero_for_indefinite",            # F63: He is doctor
            "indefinite_one",                 # F66: one book, one person

            # Tense/Aspect (eWAVE Category 3)
            "progressive_stative",            # F88: I am knowing, I am having
            "simple_present_for_perfect",     # F101: I am living here since 1990

            # Verb Morphology (eWAVE Category 5)
            "zero_3sg",                       # F170: He come daily (variable)

            # Negation (eWAVE Category 6)
            "invariant_tag",                  # F165: isn't it? You are coming, isn't it?

            # Relativization (eWAVE Category 8)
            "zero_relative_subject",          # F193: The man Ø came
            "relativizer_what",               # F191: The thing what I bought

            # Complementation (eWAVE Category 9)
            "deletion_to_infinitive",         # F209: I want Ø go

            # Discourse & Word Order (eWAVE Category 12)
            "other_fronting",                 # F224: That book, I have read
            "inverted_indirect_questions",    # F227: Tell me where is it
            "like_focus",                     # F232: He is like very smart only
            "focus_marker"                    # F225: Only he came (focus)
        ],
        "strength": 0.8,
        "notes": "Distinctive features: progressive with statives, invariant isn't it tag, topicalization, only as focus marker."
    },

    "pakistani": {
        "name": "Pakistani English",
        "region": "Pakistan",
        "description": "English as spoken in Pakistan",
        "features": [
            "progressive_stative", "zero_for_definite", "invariant_tag",
            "inverted_indirect_questions", "resumptive_pronouns"
        ],
        "strength": 0.6,
        "notes": "Similar to Indian English"
    },

    "sri_lankan": {
        "name": "Sri Lankan English",
        "region": "Sri Lanka",
        "description": "English as spoken in Sri Lanka",
        "features": [
            "progressive_stative", "zero_for_definite", "invariant_tag",
            "resumptive_pronouns", "inverted_indirect_questions"
        ],
        "strength": 0.5,
        "notes": "South Asian variety"
    },

    "singapore": {
        "name": "Colloquial Singapore English (Singlish)",
        "region": "Singapore",
        "description": "Contact variety with Chinese/Malay/Tamil substrate; distinctive sentence-final particles",
        "features": [
            # Pronouns (eWAVE Category 1)
            "object_pronoun_subject",         # F30: Him go there already
            "subject_drop_referential",       # F43: Ø Very good one

            # Noun Phrase (eWAVE Category 2)
            "zero_for_definite",              # F62: Go market
            "zero_for_indefinite",            # F63: He teacher
            "zero_plural_nonhuman",           # F58: Many book

            # Tense/Aspect (eWAVE Category 3)
            "unmarked_past",                  # F129: Yesterday I go there
            "already_perfect",                # F109: I eat already (perfective)
            "progressive_stative",            # F88: I am having

            # Verb Morphology (eWAVE Category 5)
            "zero_3sg",                       # F170: He go, she come

            # Negation (eWAVE Category 6)
            "invariant_tag",                  # F165: is it? can or not?
            "can_or_not_tag",                 # F168: Can or not? (A-not-A question)
            "negative_concord",               # F154: Don't have nothing

            # Agreement (eWAVE Category 7)
            "delete_copula_adjp",             # F177: He very tall
            "delete_copula_np",               # F176: She teacher
            "delete_copula_locative",         # F178: They at home
            "existential_get",                # F202: Got many people (existential)
            "variant_existential",            # F173: Have/got existential

            # Relativization (eWAVE Category 8)
            "zero_relative_subject",          # F193: The man Ø come
            "one_relativizer",                # F196: The one that/which

            # Complementation (eWAVE Category 9)
            "deletion_to_infinitive",         # F209: Want Ø go

            # Discourse & Word Order (eWAVE Category 12)
            "other_fronting",                 # F224: That one, very good
            "focus_marker",                   # F225: Is he do it (focus)
            "no_inversion_wh",                # F228: What you want?
            "no_inversion_yn"                 # F229: You want or not?
        ],
        "strength": 0.9,
        "notes": "Distinctive particles: lah (emphasis), leh (suggestion), lor (resignation), meh (doubt). Got as existential. Already as perfective."
    },

    "malaysian": {
        "name": "Malaysian English",
        "region": "Malaysia",
        "description": "English as spoken in Malaysia",
        "features": [
            "progressive_stative", "zero_for_definite", "invariant_tag",
            "zero_3sg", "zero_relative_subject", "already_perfect"
        ],
        "strength": 0.6,
        "notes": "Similar to Singlish but less basilectal"
    },

    "philippine": {
        "name": "Philippine English",
        "region": "Philippines",
        "description": "English as spoken in the Philippines",
        "features": [
            "progressive_stative", "zero_for_definite", "invariant_tag",
            "resumptive_pronouns", "inverted_indirect_questions"
        ],
        "strength": 0.5,
        "notes": "Tagalog substrate influence"
    },

    "hong_kong": {
        "name": "Hong Kong English",
        "region": "Hong Kong",
        "description": "English as spoken in Hong Kong",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "invariant_tag", "other_fronting", "zero_relative_subject"
        ],
        "strength": 0.6,
        "notes": "Cantonese substrate influence"
    },

    # ═══════════════════════════════════════════════════════════════
    # AUSTRALIA AND PACIFIC
    # ═══════════════════════════════════════════════════════════════

    "australian": {
        "name": "Australian English",
        "region": "Australia",
        "description": "Standard Australian English",
        "features": [
            "she_inanimate", "invariant_tag", "clause_final_but_though"
        ],
        "strength": 0.4,
        "notes": "Relatively close to British English"
    },

    "australian_vernacular": {
        "name": "Australian Vernacular English",
        "region": "Australia",
        "description": "Informal Australian English",
        "features": [
            "she_inanimate", "invariant_tag", "clause_final_but_though",
            "second_person_plural", "me_coordinate_subjects"
        ],
        "strength": 0.5,
        "notes": "More informal features"
    },

    "aboriginal": {
        "name": "Aboriginal English",
        "region": "Australia (Indigenous)",
        "description": "English of Australian Aboriginal communities with features from eWAVE documentation",
        "features": [
            # Pronouns (eWAVE Category 1)
            "alternative_it_dummy",           # F4: Alternative dummy pronoun forms
            "generalized_3sg_subject",        # F5: Generalized 3sg pronoun (em/im)
            "generalized_3sg_object",         # F6: Generalized 3sg object
            "no_gender_distinction",          # F10: No gender distinction in 3sg
            "second_person_plural",           # F34: Second person plural (youse, youfella/s)
            "me_coordinate_subjects",         # F7: Me in coordinate subjects
            "object_possessive_1sg",          # F26: Me for my

            # Noun Phrase (eWAVE Category 2)
            "plural_preposed",                # F50: Plural marking via preposed elements
            "zero_plural_human",              # F57: Optional plural marking (human)
            "zero_plural_nonhuman",           # F58: Optional plural marking (non-human)
            "zero_for_definite",              # F62: Zero article for definite
            "zero_for_indefinite",            # F63: Zero article for indefinite
            "them_for_those",                 # F68: Them for those
            "zero_genitive",                  # F77: Zero genitive (John car)

            # Tense & Aspect (eWAVE Category 3)
            "unmarked_past",                  # F129: Unmarked/base form for past
            "been_past_anterior",             # F111: Been for remote past
            "regularized_past",               # F128: Regularization (-ed extension)
            "participle_for_past",            # F131: Past participle for past tense
            "zero_past_regular",              # F132: Zero past tense forms

            # Modal Verbs (eWAVE Category 4)
            "go_future",                      # F114: Go-based future (gonna, gon)

            # Verb Morphology (eWAVE Category 5)
            "serial_verb_go",                 # F149: Serial verb go
            "serial_verb_give",               # F148: Serial verb give
            "serial_verb_come",               # F150: Serial verb come

            # Negation (eWAVE Category 6)
            "negative_concord",               # F154: Multiple negation
            "invariant_dont",                 # F158: Invariant don't
            "never_past_negator",             # F159: Never as past negator
            "invariant_tag",                  # F165: Invariant tags (eh?, ini?)

            # Agreement (eWAVE Category 7)
            "zero_3sg",                       # F170: Zero 3sg marking
            "variant_existential",            # F173: Variant existentials (e got)
            "delete_aux_progressive",         # F174: Deletion of aux be before -ing
            "delete_aux_gonna",               # F175: Deletion of aux be before gonna
            "delete_copula_np",               # F176: Deletion of copula before NP
            "delete_copula_adjp",             # F177: Deletion of copula before AdjP
            "delete_copula_locative",         # F178: Deletion of copula before locative
            "was_were_generalization",        # F180: Was/were leveling

            # Relativization (eWAVE Category 8)
            "zero_relative_subject",          # F193: Zero relativizer in subject position

            # Complementation (eWAVE Category 9)
            "deletion_to_infinitive",         # F208: Deletion of to before infinitives

            # Adverbs & Prepositions (eWAVE Category 11)
            "omit_prepositions",              # F216: Omission of prepositions

            # Discourse & Word Order (eWAVE Category 12)
            "no_inversion_wh",                # F228: No inversion in wh-questions
            "no_inversion_yn"                 # F229: No inversion in yes/no questions
        ],
        "strength": 0.9,
        "notes": "Creole-influenced variety with extensive eWAVE documentation (86 features). Key markers: copula/aux deletion, zero plural marking, invariant tags (eh?, ini?), pronoun leveling."
    },

    "kriol": {
        "name": "Roper River Creole (Kriol)",
        "region": "Australia (Northern Territory)",
        "description": "English-based creole of Northern Australia",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "been_past_anterior", "zero_plural_nonhuman", "serial_verb_go",
            "variant_existential", "transitive_suffix"
        ],
        "strength": 1.0,
        "notes": "Full creole language"
    },

    "torres_strait": {
        "name": "Torres Strait Creole",
        "region": "Australia (Torres Strait)",
        "description": "Creole of Torres Strait Islands",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "zero_plural_nonhuman", "serial_verb_go", "been_past_anterior"
        ],
        "strength": 0.9,
        "notes": "Pacific creole influence"
    },

    "new_zealand": {
        "name": "New Zealand English",
        "region": "New Zealand",
        "description": "English of New Zealand",
        "features": [
            "she_inanimate", "invariant_tag", "second_person_plural"
        ],
        "strength": 0.4,
        "notes": "Close to Australian English"
    },

    "hawaii_creole": {
        "name": "Hawaiʻi Creole (Pidgin)",
        "region": "Hawaii",
        "description": "English-based creole of Hawaii",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "stap_stay_progressive", "zero_plural_nonhuman", "for_complementizer",
            "been_past_anterior", "variant_existential"
        ],
        "strength": 0.9,
        "notes": "Distinctive 'stay' progressive"
    },

    "tok_pisin": {
        "name": "Tok Pisin",
        "region": "Papua New Guinea",
        "description": "English-based creole of Papua New Guinea",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "serial_verb_go", "zero_plural_nonhuman", "been_past_anterior",
            "variant_existential", "say_complementizer", "transitive_suffix",
            "bilong_possessive", "inclusive_exclusive_1p"
        ],
        "strength": 1.0,
        "notes": "National language of PNG"
    },

    "bislama": {
        "name": "Bislama",
        "region": "Vanuatu",
        "description": "English-based creole of Vanuatu",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "serial_verb_go", "zero_plural_nonhuman", "been_past_anterior",
            "transitive_suffix", "bilong_possessive"
        ],
        "strength": 1.0,
        "notes": "National language of Vanuatu"
    },

    "fiji": {
        "name": "Fiji English",
        "region": "Fiji",
        "description": "English as spoken in Fiji",
        "features": [
            "progressive_stative", "zero_for_definite", "zero_3sg",
            "invariant_tag"
        ],
        "strength": 0.5,
        "notes": "South Pacific variety"
    },

    "fiji_acrolectal": {
        "name": "Acrolectal Fiji English",
        "region": "Fiji",
        "ewave_id": 67,
        "description": "Educated/formal variety of Fiji English, closer to Standard English",
        "features": [
            # B-rated features from eWAVE #67
            "singular_it_for_plural",          # F41: it for they in anaphoric use
            "it_insertion",                    # F45: insertion of it where StE has zero
            "regularized_plurals",             # F48: extension of -s to irregular plurals
            "phonological_plural",             # F49: phonological regularization
            "count_mass_plural",               # F55: different count/mass distinctions
            "optional_plural_human",           # F57: plural marking optional for humans
            "optional_plural_nonhuman",        # F58: plural marking optional for non-humans
            "definite_for_indefinite",         # F60: definite article where StE has indefinite
            "indefinite_for_definite",         # F61: indefinite where StE has definite
            "definite_where_zero",             # F64: definite article where StE has zero
            "indefinite_one",                  # F66: indefinite article one/wan
            "double_comparative",              # F78: double comparatives/superlatives
            "progressive_stative",             # F88: progressive extended to stative verbs
            "past_for_perfect",                # F99: simple past for present perfect
            "perfect_for_past",                # F100: present perfect for simple past
            "present_for_perfect",             # F101: simple present for continuative perfect
            "zero_3sg",                        # F170: zero marking for 3rd person singular
            "invariant_tag",                   # F165: invariant non-concord tags
            "inverted_indirect_questions",     # F227: inverted word order in indirect questions
            "no_inversion_yesno"               # F229: no inversion in yes/no questions
        ],
        "strength": 0.4,
        "notes": "Formal/educated Fiji English with minimal distinctive features (eWAVE #67)"
    },

    "fiji_basilectal": {
        "name": "Pure Fiji English (Basilectal)",
        "region": "Fiji",
        "ewave_id": 68,
        "description": "Colloquial/basilectal Fiji English with extensive dialectal features",
        "features": [
            # A-rated features from eWAVE #68 (pervasive/obligatory)
            "alternative_it",                  # F3: alternative forms for referential it
            "generalized_3sg_subject",         # F5: generalized 3sg subject pronouns
            "generalized_3sg_object",          # F6: generalized 3sg object pronouns
            "me_coordinate_subjects",          # F7: me instead of I in coordinate subjects
            "myself_coordinate",               # F8: myself in coordinate subjects
            "no_gender_3sg",                   # F10: no gender distinction in 3sg
            "subject_pronoun_possessive_1pl",  # F19: subject pronoun as possessive 1pl
            "subject_pronoun_possessive_3pl",  # F21: subject pronoun as possessive 3pl
            "second_person_possessive_other",  # F23: 2nd person possessive other than you
            "object_pronoun_possessive_3pl",   # F25: object pronoun as possessive 3pl
            "object_pronoun_possessive_1pl",   # F27: object pronoun as possessive 1pl
            "us_np_subject",                   # F28: us + NP in subject function
            "subject_in_object_function",      # F30: subject pronoun in object function
            "object_in_subject_function",      # F31: object pronoun in subject function
            "emphatic_nonemphatic_pronouns",   # F32: distinction emphatic vs non-emphatic
            "second_person_plural",            # F34: 2nd person plural other than you
            "number_distinctions_pronouns",    # F37: more number distinctions in pronouns
            "specialized_plural_markers",      # F38: specialized plural markers for pronouns
            "object_pronoun_drop",             # F42: object pronoun drop
            "phonological_plural",             # F49: phonological regularization of plurals
            "postposed_plural_elements",       # F51: plural marking via postposed elements
            "associative_plural_other",        # F53: associative plural by other elements
            "count_mass_plural",               # F55: different count/mass distinctions
            "definite_for_indefinite",         # F60: definite for indefinite article
            "indefinite_for_definite",         # F61: indefinite for definite article
            "zero_for_definite",               # F62: zero article for definite
            "zero_for_indefinite",             # F63: zero article for indefinite
            "definite_where_zero",             # F64: definite where StE has zero
            "indefinite_one",                  # F66: indefinite article one/wan
            "demonstrative_for_definite",      # F67: demonstratives for definite articles
            "them_for_those",                  # F68: them instead of those
            "double_comparative",              # F78: double comparatives/superlatives
            "analytic_comparison",             # F80: extension of analytic comparison
            "progressive_stative",             # F88: progressive extended to stative verbs
            "resultative_there",               # F96: there with past participle
            "been_past_anterior",              # F111: past tense/anterior marker been
            "loosened_sequence_of_tenses",     # F113: loosening of sequence of tenses
            "go_future",                       # F114: go-based future markers
            "present_for_future",              # F117: present tense for neutral future
            "participle_for_past",             # F131: past participle for past tense
            "zero_past",                       # F132: zero past tense for regular verbs
            "was_for_were",                    # F147: was for conditional were
            "never_past_negator",              # F159: never as preverbal past negator
            "invariant_tag",                   # F165: invariant non-concord tags
            "non_standard_negative_response",  # F169: non-standard negative responses
            "zero_3sg",                        # F170: zero 3sg marking
            "delete_aux_progressive",          # F174: deletion of aux be before progressive
            "delete_aux_gonna",                # F175: deletion of aux be before gonna
            "delete_copula_np",                # F176: copula deletion before NPs
            "delete_copula_adjp",              # F177: copula deletion before AdjPs
            "delete_copula_locative",          # F178: copula deletion before locatives
            "delete_aux_have",                 # F179: deletion of auxiliary have
            "resumptive_pronouns",             # F194: resumptive/shadow pronouns
            "deletion_of_to",                  # F208: deletion of to before infinitives
            "flat_adverbs",                    # F220/221: adverbs same form as adjectives
            "no_inversion_wh",                 # F228: no inversion in wh-questions
            "no_inversion_yesno",              # F229: no inversion in yes/no questions
            "subject_in_imperatives",          # F233: presence of subject in imperatives
            "like_quotative"                   # F235: like as quotative particle
        ],
        "strength": 1.0,
        "notes": "Basilectal Fiji English with extensive substrate influence (eWAVE #68)"
    },

    "solomon_islands_pijin": {
        "name": "Solomon Islands Pijin",
        "region": "Solomon Islands",
        "description": "English-based creole of Solomon Islands",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "serial_verb_go", "zero_plural_nonhuman", "been_past_anterior",
            "transitive_suffix", "bilong_possessive", "inclusive_exclusive_1p"
        ],
        "strength": 1.0,
        "notes": "Melanesian Pidgin variety"
    },

    # ═══════════════════════════════════════════════════════════════
    # ATLANTIC ISLANDS
    # ═══════════════════════════════════════════════════════════════

    "falkland": {
        "name": "Falkland Islands English",
        "region": "Falkland Islands",
        "description": "English of the Falkland Islands",
        "features": [
            "she_inanimate", "regularized_reflexives"
        ],
        "strength": 0.3,
        "notes": "British-influenced"
    },

    "tristan": {
        "name": "Tristan da Cunha English",
        "region": "Tristan da Cunha",
        "description": "English of Tristan da Cunha",
        "features": [
            "regularized_reflexives", "for_to_infinitive", "double_comparatives",
            "zero_3sg", "was_were_generalization", "negative_concord"
        ],
        "strength": 0.6,
        "notes": "Isolated dialect with archaic features"
    },

    "st_helena": {
        "name": "St. Helena English",
        "region": "St. Helena",
        "description": "English of St. Helena",
        "features": [
            "regularized_reflexives", "for_to_infinitive", "zero_3sg",
            "negative_concord"
        ],
        "strength": 0.5,
        "notes": "South Atlantic variety"
    },

    "pitcairn_norfolk": {
        "name": "Norfolk Island/Pitcairn English",
        "region": "Norfolk Island / Pitcairn Islands",
        "description": "Creoloid of Norfolk and Pitcairn Islands",
        "features": [
            "regularized_reflexives", "zero_3sg", "unmarked_past",
            "zero_plural_nonhuman"
        ],
        "strength": 0.7,
        "notes": "Bounty mutineer descendants"
    },

    # ═══════════════════════════════════════════════════════════════
    # ADDITIONAL eWAVE VARIETIES
    # ═══════════════════════════════════════════════════════════════

    "acrolectal_southeast_caribbean": {
        "name": "Acrolectal Southeast Caribbean English",
        "region": "Southeast Caribbean",
        "description": "Educated English of Southeast Caribbean islands",
        "features": [
            "progressive_stative", "invariant_tag", "zero_3sg"
        ],
        "strength": 0.4,
        "notes": "More formal register Caribbean English"
    },

    "saramaccan": {
        "name": "Saramaccan",
        "region": "Suriname",
        "description": "English-Portuguese based creole of Suriname Maroons",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "serial_verb_go", "serial_verb_give", "serial_verb_come",
            "zero_plural_nonhuman", "focus_marker"
        ],
        "strength": 1.0,
        "notes": "Deep creole with Portuguese elements"
    },

    "krio": {
        "name": "Krio (Sierra Leone)",
        "region": "Sierra Leone",
        "description": "English-based creole of Sierra Leone",
        "features": [
            "delete_copula_adjp", "zero_3sg", "unmarked_past",
            "serial_verb_go", "say_complementizer", "zero_plural_nonhuman",
            "focus_marker", "variant_existential"
        ],
        "strength": 1.0,
        "notes": "Major West African creole"
    },

    "cape_flats": {
        "name": "Cape Flats English",
        "region": "South Africa (Western Cape)",
        "description": "English of Cape Flats communities",
        "features": [
            "progressive_stative", "invariant_tag", "negative_concord",
            "resumptive_pronouns"
        ],
        "strength": 0.6,
        "notes": "Afrikaans and Cape Malay influence"
    }
}

# Region groupings for easier access
REGIONS = {
    "north_america": [
        "urban_aave", "rural_aave", "earlier_aave", "appalachian", "ozark",
        "southeast_enclave", "chicano", "gullah", "newfoundland", "colloquial_american"
    ],
    "british_isles": [
        "scottish", "irish", "welsh", "northern_england", "southwest_england",
        "east_anglian", "southeast_england", "orkney_shetland", "manx",
        "channel_islands", "british_creole", "maltese_english"
    ],
    "caribbean": [
        "jamaican", "bahamian", "trinidadian", "guyanese", "barbadian",
        "belizean", "vincentian", "acrolectal_southeast_caribbean", "saramaccan"
    ],
    "africa": [
        "nigerian", "nigerian_pidgin", "ghanaian", "cameroon", "kenyan",
        "tanzanian", "ugandan", "liberian_settler", "south_african_black",
        "south_african_indian", "south_african_white", "zimbabwean_white",
        "krio", "cape_flats"
    ],
    "south_southeast_asia": [
        "indian", "pakistani", "sri_lankan", "singapore", "malaysian",
        "philippine", "hong_kong"
    ],
    "australia_pacific": [
        "australian", "australian_vernacular", "aboriginal", "kriol",
        "torres_strait", "new_zealand", "hawaii_creole", "tok_pisin",
        "bislama", "fiji", "solomon_islands_pijin"
    ],
    "atlantic_islands": [
        "falkland", "tristan", "st_helena", "pitcairn_norfolk"
    ]
}


def get_dialect(dialect_key: str) -> dict:
    """Get dialect configuration by key."""
    if dialect_key not in DIALECT_REGISTRY:
        raise ValueError(f"Unknown dialect: {dialect_key}. Available: {list(DIALECT_REGISTRY.keys())}")
    return DIALECT_REGISTRY[dialect_key]


def list_dialects_by_region(region: str = None) -> list:
    """List dialects, optionally filtered by region."""
    if region:
        if region not in REGIONS:
            raise ValueError(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")
        return [(key, DIALECT_REGISTRY[key]["name"]) for key in REGIONS[region] if key in DIALECT_REGISTRY]
    return [(key, d["name"]) for key, d in DIALECT_REGISTRY.items()]


def get_all_features_for_dialect(dialect_key: str) -> list:
    """Get all features associated with a dialect."""
    dialect = get_dialect(dialect_key)
    return dialect.get("features", [])


def get_dialect_count() -> int:
    """Get total number of dialects."""
    return len(DIALECT_REGISTRY)


def validate_dialect_features(dialect_key: str, feature_library: dict) -> list:
    """Validate that all features in a dialect exist in the feature library.

    Returns list of invalid feature keys.
    """
    dialect = get_dialect(dialect_key)
    features = dialect.get("features", [])
    invalid = [f for f in features if f not in feature_library]
    return invalid


# ═══════════════════════════════════════════════════════════════════════════
# RATING-BASED FEATURE SELECTION
# Uses eWAVE scraped data for comprehensive, rating-based feature selection
# ═══════════════════════════════════════════════════════════════════════════

def get_features_by_rating(dialect_key: str, rating_level: str = "AB") -> list:
    """
    Get features for a dialect based on eWAVE ratings.

    This uses the scraped eWAVE data to provide accurate, rating-based
    feature selection instead of manually curated feature lists.

    Args:
        dialect_key: Dialect identifier (e.g., "aboriginal", "urban_aave")
        rating_level: Which ratings to include:
            - "A": Only pervasive/obligatory features (most conservative)
            - "AB": Pervasive + common features (recommended default)
            - "ABC": All documented features including rare ones
            - "ABCD": Include attested absences (for analysis)

    Returns:
        List of feature keys

    Example:
        >>> # Get only highly characteristic features
        >>> features = get_features_by_rating("aboriginal", "A")
        >>> len(features)
        51

        >>> # Get common features (recommended)
        >>> features = get_features_by_rating("aboriginal", "AB")
        >>> len(features)
        84
    """
    try:
        from ewave_ratings import get_features_by_rating as _get_features, RatingLevel
        level = RatingLevel(rating_level)
        return _get_features(dialect_key, level)
    except ImportError:
        # Fallback to manually defined features if ewave_ratings not available
        return get_all_features_for_dialect(dialect_key)
    except ValueError:
        # Fallback if dialect not in eWAVE data
        return get_all_features_for_dialect(dialect_key)


def get_dialect_with_rating(dialect_key: str, rating_level: str = "AB") -> dict:
    """
    Get dialect configuration with features filtered by eWAVE rating.

    This returns a copy of the dialect config with the features list
    replaced by rating-filtered features from eWAVE data.

    Args:
        dialect_key: Dialect identifier
        rating_level: Rating level ("A", "AB", "ABC", "ABCD")

    Returns:
        Dialect configuration dictionary with filtered features
    """
    dialect = get_dialect(dialect_key).copy()
    try:
        features = get_features_by_rating(dialect_key, rating_level)
        dialect["features"] = features
        dialect["rating_level"] = rating_level
        dialect["feature_source"] = "ewave"
    except Exception:
        dialect["rating_level"] = "manual"
        dialect["feature_source"] = "manual"
    return dialect


def get_rating_summary(dialect_key: str) -> dict:
    """
    Get a summary of feature ratings for a dialect.

    Returns:
        Dictionary with rating counts and feature counts per level
    """
    try:
        from ewave_ratings import get_dialect_rating_summary
        return get_dialect_rating_summary(dialect_key)
    except ImportError:
        return {"error": "ewave_ratings module not available"}
    except ValueError as e:
        return {"error": str(e)}


def compare_dialect_features(dialect_keys: list, rating_level: str = "AB") -> dict:
    """
    Compare features across multiple dialects at a given rating level.

    Args:
        dialect_keys: List of dialect identifiers to compare
        rating_level: Rating level for comparison

    Returns:
        Dictionary with shared and unique features
    """
    try:
        from ewave_ratings import compare_dialects, RatingLevel
        level = RatingLevel(rating_level)
        return compare_dialects(dialect_keys, level)
    except ImportError:
        # Manual comparison fallback
        all_features = {d: set(get_all_features_for_dialect(d)) for d in dialect_keys}
        shared = set.intersection(*all_features.values()) if all_features else set()
        unique = {d: f - shared for d, f in all_features.items()}
        return {
            "rating_level": rating_level,
            "dialects": dialect_keys,
            "shared_features": list(shared),
            "shared_count": len(shared),
            "unique_features": {k: list(v) for k, v in unique.items()},
            "per_dialect_count": {k: len(v) for k, v in all_features.items()}
        }
