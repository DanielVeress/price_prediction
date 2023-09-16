def format_num(num, infix="'", postfix='$'):
    reversed_num = str(num)[::-1]
    formatted_num = ''
    grouping = 3 # digits
    
    group_idx = 0
    for letter in reversed_num:
        if letter.isdigit():
            if group_idx == grouping:
                formatted_num = infix + formatted_num
                group_idx = 0
            formatted_num = letter + formatted_num
            group_idx += 1
        else:
            formatted_num = letter + formatted_num
    
    return formatted_num + postfix