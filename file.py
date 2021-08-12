some_text = 'abcabcabc'

def first_not_duplicate_char(txt):
    has_no_duplicate_yet = []
    has_duplicates = []
    for char in txt:
        if char in has_no_duplicate_yet:
            if char in has_duplicates:
                continue
            else:
                has_duplicates.append(char)
        else:
            has_no_duplicate_yet.append(char)
    
    final_no_dup_list = [x for x in has_no_duplicate_yet if x not in has_duplicates]

    if len(final_no_dup_list) == 0:
        return '_'
    else:
        return final_no_dup_list[0]


print(first_not_duplicate_char(some_text))
