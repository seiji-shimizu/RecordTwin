def generate_example(text, entity):
    
    text_lines = text.split("\n")
    entity_lines = entity.split("\n")
    
    text_prompt = ""
    entity_prompt = ""

   
    num_lines = len(text_lines)
    entity_prompt +='The number of lines: ' + str(num_lines) + '\n'

    entity_prompt += 'lists of entities: \n'

    text_prompt += 'Generated lines: \n'

    for i, (text, entities) in enumerate(zip(text_lines, entity_lines)):
        text_prompt += f"{i+1}| {text}|\n"
        
        # if there is , at the begining of entities, remove it
        if entities:
            # if there is , at the begining of entities, remove it
            if entities[0] == ',':
                entities = entities[1:]
            entity_prompt += f"{i+1}| {entities}|\n"
            
        else:
            entity_prompt += f"{i+1}||\n"
         
    
    return text_prompt, entity_prompt


def generate_entities(entity):

    entity_lines = entity.split("\n")
    entity_prompt = ""
    entity_prompt += '# Generate from this list of entities: \n'
    entity_prompt += 'The number of linses: ' + str(len(entity_lines)) + '\n'
    
    entity_prompt += 'lists of entities: \n'
    # remove empty lines
    #entity_lines = [entity for entity in entity_lines if entity]
    
    for i, entity in enumerate(entity_lines):
        # if entity is not empty
        if entity:
            # if there is , at the begining of entities, remove it
            if entity[0] == ',':
                entity = entity[1:]
            entity_prompt += f"{i+1}| {entity}|\n"

        else:
            entity_prompt += f"{i+1}| |\n"

    entity_prompt += '\n\n' + 'Generated lines: \n1|'
    return entity_prompt


def generate_prompt(template, data, entities):
    
    examples = ''
    entities = generate_entities(entities)
    
    for i, sample in enumerate(data):
        text = sample['text']
        entity = sample['entities']

        examples += f"Example{i+1}\n"
        text_prompt, entity_prompt = generate_example(text, entity)
        if entity_prompt:
            examples +=  entity_prompt + "\n\n" + text_prompt + "\n\n"
        else:
            examples += text_prompt + "\n\n"

    template = template.replace("[examples]", examples)
    template = template.replace("[entities]", entities)

    return template