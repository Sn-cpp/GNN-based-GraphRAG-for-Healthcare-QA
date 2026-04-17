class MistralPrompt:
    def get_refining_prompt(triplets: list[tuple, ]):
        """
        Helper function to build the refining prompt for triplets 
        """
        
        triplet_str = "\n".join([f"({s}, {r}, {o})" for s, r, o in triplets])
        return f"""[INST]
        You normalize relations in knowledge graph triplets into a canonical schema.

        Input:
        {triplet_str}

        Rules:
        - Keep subject and object EXACTLY unchanged
        - Rewrite ONLY the relation
        - Use lowercase snake_case
        - Remove filler words (of, the, type, kind, class)

        CANONICAL RELATIONS (MANDATORY):
        Map every relation to ONE of:

        - studies        (X studies Y)
        - is_a           (X is a type of Y)
        - part_of        (X is part of Y)
        - has_part       (X has part Y)
        - practices      (X practices Y)

        DIRECTION RULES:
        - Always use ACTIVE voice
        - Convert and reverse when needed:
        studied_by / is_studied_by / are_studied_by → studies (REVERSE)
        is_part_of / are_parts_of → part_of
        has_parts / have_parts → has_part
        is_subclass_of / has_subclass → is_a
        is_practiced_by → practices (REVERSE)

        - Enforce canonical direction:
        (anatomy, studies, structures) ✓
        (structures, studied_by, anatomy) ✗

        CONSISTENCY:
        - Do NOT output both directions of the same fact
        - Prefer the canonical direction above

        VALIDITY:
        - If subject == object → DROP the triplet
        - If relation cannot be mapped → DROP the triplet
        - If subject or object is not an entity → DROP the triplet

        Output:
    - ONE triplet per line
    - Format: (subject, relation, object)
    - No extra text
    [/INST]"""