def makeTable(headerRow, columnizedData, columnSpacing=2):
    from numpy import array, max, vectorize

    cols = array(columnizedData, dtype=str)
    colSizes = [max(vectorize(len)(col)) for col in cols]

    header = ''
    rows = ['' for i in cols[0]]

    for i in range(0, len(headerRow)):
        if len(headerRow[i]) > colSizes[i]: colSizes[i] = len(headerRow[i])
        headerRow[i] += ' ' * (colSizes[i] - len(headerRow[i]))
        header += headerRow[i]
        if not i == len(headerRow) - 1: header += ' ' * columnSpacing

        for j in range(0, len(cols[i])):
            if len(cols[i][j]) < colSizes[i]:
                cols[i][j] += ' ' * (colSizes[i] - len(cols[i][j]) + columnSpacing)
            rows[j] += cols[i][j]
            if not i == len(headerRow) - 1: rows[j] += ' ' * columnSpacing

    line = '-' * len(header)
    print(line)
    print(header)
    print(line)
    for row in rows: print(row)
    print(line)
