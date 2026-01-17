from propella import get_annotation_response_schema

schema = get_annotation_response_schema(as_string=True, compact_whitespace=True)

with open("propella_schema.txt", "w") as f:
    f.write(schema)
