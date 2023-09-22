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


def format_backtest_output(output, wanted_metrics):
    dict_output = dict(output)

    symbols_to_replace = {
        '[%]': 'PREC',
        '[$]': 'USD',
        '&': 'and',
        '#': 'Num of',
        '(Ann.)': '/Annual/'
    }

    filtered_output = {}
    for key, value in dict_output.items():
        
        # only save  parameters, that are wanted
        if key in wanted_metrics:
            
            # replace not allowed symbols
            new_key = key
            for symbol, replacement in symbols_to_replace.items():
                if symbol in new_key:
                    new_key = new_key.replace(symbol, replacement)
            
            filtered_output[new_key] = value
    
    return filtered_output