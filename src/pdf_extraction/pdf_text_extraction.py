#%%
import sys
sys.path.append('../../')
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LAParams, LTRect, LTTextBoxHorizontal, LTTextBoxVertical
import re
from statistics import mode
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
import camelot

def text_extraction(element, page_num):
    """
    Extracts text information from a PDF element.

    Args:
        element (object): The PDF element from which text information is extracted.
        page_num (int): The page number of the PDF.

    Returns:
        dict or None: A dictionary containing extracted text information if valid, otherwise None.

    Note:
        This function extracts text information from a PDF element. It processes the text to handle encoding errors
        and removes unwanted characters. It also calculates the overall font weight and font size of the text.
        The function returns a dictionary containing the extracted text, font weight, font size, the original element,
        and the page number.

    Example:
        >>> element = ...  # PDF element
        >>> page_num = 1
        >>> extracted_text = text_extraction(element, page_num)
    """
    line_text = element.get_text()
    line_text = line_text.encode('utf-8', errors='replace').decode(encoding='utf-8', errors='ignore').replace('\x00', '').replace('�', 'ti')

    if re.match(r'(\s*\n\s*)', line_text) or re.fullmatch(r'([a-zA-Z]\s)+', line_text):
        return None

    char_font_weights = []
    char_font_sizes = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for char in text_line:
                if isinstance(char, LTChar):
                    char_font_weights.append('bold' if 'Bold' in char.fontname else 'normal')
                    char_font_sizes.append(round(char.size))

    overall_font_weight = mode(char_font_weights)   
    overall_font_size = mode(char_font_sizes)

    line_text = line_text.replace('\n', '').strip()

    return {
        "text" : line_text,
        "font_weight" : overall_font_weight,
        "font_size" : overall_font_size,
        "element" : element,
        "page": page_num,
    }

def get_pages_properties(pages):
    """
    Extracts properties from a list of PDF pages.

    Args:
        pages (list): A list of PDF pages.

    Returns:
        tuple: A tuple containing font sizes (list), text spaces (list), and page height (float).

    Note:
        This function extracts properties from a list of PDF pages. It iterates over each page and extracts text
        properties such as font sizes and text spaces. It calculates the page height based on the bounding box
        of the first page. The function returns a tuple containing font sizes, text spaces, and page height.

    Example:
        >>> pages = [...]  # List of PDF pages
        >>> font_sizes, text_spaces, page_height = get_pages_properties(pages)
    """
    document = []
    font_sizes = []
    text_spaces = []

    for i, page in enumerate(pages):
        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda x: x[0], reverse=True)

        for _, element in page_elements:
            if isinstance(element, LTTextContainer):
                line = text_extraction(element, i)
                if line:
                    if len(document) > 0:
                        enter_height = abs(round(document[-1]['element'].bbox[1] - line['element'].bbox[1]))
                        text_spaces.append(enter_height)
                    
                    document.append(line)
                    font_sizes.append(line['font_size'])

        page_bbox = pages[0].bbox
        page_height = page_bbox[3] - page_bbox[1]

    return font_sizes, text_spaces, page_height

def bbox_intersect(ba, bb):
    """
    Checks if two bounding boxes intersect.

    Args:
        ba (tuple): Bounding box coordinates of the first rectangle in the format (x1, y1, x2, y2).
        bb (tuple): Bounding box coordinates of the second rectangle in the format (x1, y1, x2, y2).

    Returns:
        bool: True if the bounding boxes intersect, False otherwise.

    Note:
        This function checks whether two bounding boxes intersect with each other. It takes two tuples,
        representing the coordinates of the top-left and bottom-right corners of each bounding box.
        The function returns True if the bounding boxes intersect, otherwise False.

    Example:
        >>> bbox1 = (0, 0, 10, 10)
        >>> bbox2 = (5, 5, 15, 15)
        >>> intersection = bbox_intersect(bbox1, bbox2)
    """
    return ba[2] >= bb[0] and bb[2] >= ba[0] and ba[3] >= bb[1] and bb[3] >= ba[1]

def bbox_next_position(ba, bb):
    """
    Determines if one bounding box is positioned below another.

    Args:
        ba (tuple): Bounding box coordinates of the first rectangle in the format (x1, y1, x2, y2).
        bb (tuple): Bounding box coordinates of the second rectangle in the format (x1, y1, x2, y2).

    Returns:
        bool: True if the first bounding box is positioned below the second, False otherwise.

    Note:
        This function compares the y-coordinates of two bounding boxes to determine if the first bounding box
        is positioned below the second. It takes two tuples, representing the coordinates of the top-left and
        bottom-right corners of each bounding box. The function returns True if the first bounding box is
        positioned below the second, otherwise False.

    Example:
        >>> bbox1 = (0, 0, 10, 10)
        >>> bbox2 = (0, 20, 10, 30)
        >>> below = bbox_next_position(bbox1, bbox2)
    """
    return ba[1] < bb[1]

def preprocess_table_df(df):
    """
    Preprocesses a DataFrame representing a table.

    This function prepares the DataFrame for further processing by joining the column names and
    values into text blocks.

    Args:
        df (DataFrame): DataFrame representing the table.

    Returns:
        list of str: A list containing the preprocessed text blocks.

    Note:
        This function expects the first row of the DataFrame to contain column names.
        It joins the column names and values into text blocks, separating each column with a slash ('/').
        Any newlines ('\n'), null characters ('\x00'), or replacement characters ('�') found in the values are removed.
        The resulting text blocks are stored in a list and returned.

    Example:
        >>> import pandas as pd
        >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        >>> df = pd.DataFrame(data)
        >>> blocks = preprocess_table_df(df)
    """
    block_text = []

    columns = df.iloc[0, :]
    df.columns = columns
    df = df.iloc[1:, :]

    block_text.append(' / '.join(df.columns).replace('\n', ' '))
    block_text.extend([' '.join(l).replace('\n', ' ').replace('\x00', 'ti').replace('�', 'ti').replace('.', '') + '.' for l in df.values])

    return block_text

def extract_text_blocks(pages, pdf_path, normal_text_spacing):
    """
    Extracts text blocks from PDF pages, handling tables and normal text.

    Args:
        pages (list of LTPage): List of PDF pages, represented as LTPage objects.
        pdf_path (str): Path to the PDF file.
        normal_text_spacing (int): Normal spacing between lines of text.

    Returns:
        list of list of dict: List of text blocks, where each block is represented as a list of dictionaries.
            Each dictionary contains information about a text line, including its text, font weight, font size, 
            associated element (from PDF), and page number.

    Note:
        This function iterates over each page of the PDF, extracting text lines.
        It handles tables by detecting table elements and substituting them with their content extracted using Camelot.
        The extracted text lines are organized into blocks, where each block represents a cohesive unit of text.
        Blocks are separated based on the normal spacing between lines of text.

    Example:
        >>> from pdfminer.high_level import extract_pages
        >>> pdf_path = "example.pdf"
        >>> with open(pdf_path, "rb") as f:
        >>>     pages = list(extract_pages(pdf_path))
        >>>     blocks = extract_text_blocks(pages, pdf_path, normal_text_spacing=10)
    """
    blocks = []
    block_text = []
    section_tables = []

    for i, page in enumerate(pages):
        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda x: x[0], reverse=True)
        tables = camelot.read_pdf(pdf_path, pages=f'{page.pageid}', iterations=2)
        table_coords = None
        prev_table_element = None

        if tables:
            # start and end position in y coordinate (hight)
            table_coords = [(table._bbox, table.df) for table in tables]

        for _, element in page_elements:
            if isinstance(element, LTTextContainer):
                # looking for the same element as the table to subtitute to the content of the extracted from pandas
                if table_coords and any(bbox_intersect(element.bbox, box) for box, _ in table_coords):
                    mask = [bbox_intersect(element.bbox, box) for box, _ in table_coords]
                    table_index = np.where(mask)[0][0]
                    df = table_coords[table_index][1]

                    if len(section_tables) > 0 and not section_tables[-1].equals(df):
                        section_tables.append(df)
                    elif len(section_tables) == 0:
                        section_tables.append(df)

                    prev_table_element = element

                    del table_coords[table_index]
                    continue
                
                # sometimes tables are detect as LTRect by pdfminer
                # it's required to calculate the next position belonging to the table
                elif table_coords:
                    for j, table in enumerate(table_coords):
                        if bbox_next_position(element.bbox, table[0]):
                            df = table[1]
                            if len(section_tables) > 0 and not section_tables[-1].equals(df):
                                section_tables.append(df)
                            elif len(section_tables) == 0:
                                section_tables.append(df)
                            prev_table_element = element
                            del table_coords[j]
                
                # appent the table text to the block of text
                if prev_table_element:
                    if len(block_text) > 0:
                        blocks.append(block_text)
                        block_text = []
                    for t in section_tables:
                        df_texts = preprocess_table_df(t)
                        for text in df_texts:
                            blocks.append([
                                {
                                    "text" : text,
                                    "font_weight" : 'normal',
                                    "font_size" : 20,
                                    "element" : prev_table_element,
                                    "page": i,
                                }
                            ])
                    prev_table_element = None
                    section_tables = []

                line = text_extraction(element, i)
                if line:
                    if len(block_text) > 0:
                        text_line_height = abs(round(block_text[-1]['element'].bbox[1] - line['element'].bbox[1]))
                        if text_line_height > normal_text_spacing:
                            # print('ENTER SPACING****')
                            blocks.append(block_text)
                            block_text = []
                    block_text.append(line)
    return blocks
        
def is_title_section(line, max_words=10):
    """
    Determines if a given text line represents a title section based on its text content and font weight.

    Args:
        line (dict): Information about the text line, including its text, font weight, and other properties.
        max_words (int): Maximum number of words allowed in the title.

    Returns:
        bool: True if the text line is considered a title section, False otherwise.

    Note:
        This function checks if the text line:
        - Contains a limited number of words (controlled by max_words).
        - Has bold font weight.
        - Matches a specific title pattern using regular expressions.

    Example:
        >>> line = {
        >>>     "text": "1. Introduction",
        >>>     "font_weight": "bold",
        >>>     # Other properties...
        >>> }
        >>> is_title = is_title_section(line, max_words=5)
    """
    title_pattern = re.compile(r'((\d{1,2}(?: ‒)?\.?)\s*[^\)]([a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑﬀ/\(\)-][\s,:.\u2013]*)+)')    
    text = line['text'].strip()
    text_length = text.split()

    if len(text_length) <= max_words and line['font_weight'] == 'bold':
        if title_pattern.fullmatch(text):
            return True
    return False

def is_title_text_propriety(line, min_font_size, max_words=10):
    """
    Determines if a given text line meets certain criteria to be considered a title.

    Args:
        line (dict): Information about the text line, including its text, font weight, font size, and other properties.
        min_font_size (int): Minimum font size allowed for the text line to be considered a title.
        max_words (int): Maximum number of words allowed in the title.

    Returns:
        bool: True if the text line meets the criteria to be considered a title, False otherwise.

    Note:
        This function checks if the text line:
        - Has bold font weight.
        - Has a font size greater than or equal to the specified minimum font size.
        - Contains a limited number of words (controlled by max_words).

    Example:
        >>> line = {
        >>>     "text": "Introduction",
        >>>     "font_weight": "bold",
        >>>     "font_size": 12,
        >>>     # Other properties...
        >>> }
        >>> is_title = is_title_text_propriety(line, min_font_size=10, max_words=5)
    """
    if (line['font_weight'] == 'bold' and line['font_size'] >= min_font_size) and len(line['text'].strip().split()) <= max_words:
        return True
    return False
    
def calculate_distance(box1, box2):
    """
    Calculate the Euclidean distance between the centers of two text boxes.

    Args:
        box1 (LTTextBox): First text box represented as an LTTextBox object.
        box2 (LTTextBox): Second text box represented as an LTTextBox object.

    Returns:
        float: The Euclidean distance between the centers of the two text boxes.

    Note:
        This function calculates the distance between the centers of two text boxes.
        It uses the Euclidean distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2).

    Example:
        >>> from pdfminer.layout import LTTextBox
        >>> box1 = LTTextBox((x0, y0, x1, y1))
        >>> box2 = LTTextBox((x0, y0, x1, y1))
        >>> distance = calculate_distance(box1, box2)
    """
    center1 = ((box1.x0 + box1.x1) / 2, (box1.y0 + box1.y1) / 2)
    center2 = ((box2.x0 + box2.x1) / 2, (box2.y0 + box2.y1) / 2)
    
    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    return distance

def text_block_categorizer(elements, min_sample, eps):
    """
    Perform text block categorization using the DBSCAN clustering algorithm. This is usefull to get footer and 
    header contents.

    Args:
        elements (list): A list of tuples containing text block elements and their corresponding page numbers.
                         Each tuple is of the form (element, page_num), where 'element' is a text block element
                         and 'page_num' is the page number where the element appears.
        min_sample (int): The number of samples in a neighborhood for a point to be considered as a core point.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.

    Returns:
        list: A list of binary labels indicating the category (0 or 1) for each text block element.

    Note:
        This function categorizes text block elements into two categories using the DBSCAN clustering algorithm.
        It considers elements in the same cluster as category 0, while elements in other clusters are labeled as category 1.
        The DBSCAN algorithm is used to group text block elements based on their spatial proximity.

    Example:
        >>> elements = [(element1, page_num1), (element2, page_num2), ...]
        >>> min_sample = 5
        >>> eps = 2.0
        >>> labels = text_block_categorizer(elements, min_sample, eps)
    """
    X = np.array([(*ele['element'].bbox, len(ele['text'])) for ele, _ in elements])

    dbscan = DBSCAN(min_samples=min_sample, eps=eps)
    dbscan.fit(X)
    labels = dbscan.labels_
    # n_clusters = len(np.unique(labels))
    label_counter = Counter(labels)
    most_common_label = label_counter.most_common(1)[0][0]
    labels = [0 if label == most_common_label else 1 for label in labels]
    labels = labels

    # print(f"{n_clusters} clusters for {len(elements)} elements")

    return labels

def is_footer(line, page_height, font_size, footer_factor):
    """
    Determine if a text line is part of the footer based on its position and font size.

    Args:
        line (dict): A dictionary representing a text line containing information such as text, font weight, font size, etc.
        page_height (float): The height of the page where the text line appears.
        font_size (float): The threshold font size for considering a line as part of the footer.
        footer_factor (float): The fraction of the page height that determines the footer's position.

    Returns:
        bool: True if the line is part of the footer, False otherwise.

    Note:
        This function checks whether a text line is part of the footer based on its vertical position and font size.
        If the line's vertical position is below a certain fraction of the page height or its font size is below a specified threshold,
        it is considered part of the footer.

    Example:
        >>> line = {'text': 'Copyright 2024', 'element': LTTextLineHorizontal(...), 'font_size': 10.0, ...}
        >>> page_height = 792.0
        >>> font_size = 8.0
        >>> footer_factor = 0.9
        >>> is_footer(line, page_height, font_size, footer_factor)
        True
    """
    if line['element'].bbox[1] < page_height * footer_factor or line['font_size'] < font_size:
        return True
    return False

def remove_header_footer(blocks, page_height, footer_factor, font_size, min_sample=3, eps=0.9):
    """
    Remove header and footer text blocks from the document blocks list based on categorization.

    Args:
        blocks (list): A list of text blocks, where each block is represented as a list of dictionaries containing text, font weight, font size, etc.
        page_height (float): The height of the page where the text blocks appear.
        footer_factor (float): The fraction of the page height that determines the footer's position.
        font_size (float): The threshold font size for considering a line as part of the footer.
        min_sample (int, optional): The minimum number of samples in a cluster required for a text block to be categorized as a header or footer. Defaults to 3.
        eps (float, optional): The maximum distance between two samples for one to be considered as part of the same neighborhood. Defaults to 0.9.

    Returns:
        None

    Note:
        This function removes text blocks identified as headers or footers from the document blocks list based on the output labels
        from the text block categorizer function. It considers text blocks with font size smaller than a threshold, as well as those
        located at the top or bottom of the page, as headers or footers. Additionally, it ignores blocks containing specific keywords
        such as "edital," "chamada," "seleção," or "programa" to avoid removing essential information.

    Example:
        >>> blocks = [
        ...     [{'text': 'Header', 'font_size': 14.0, ...}, {'text': 'Content', 'font_size': 12.0, ...}, ...],
        ...     [{'text': 'Content', 'font_size': 12.0, ...}, {'text': 'Footer', 'font_size': 14.0, ...}, ...],
        ...     ...
        ... ]
        >>> page_height = 792.0
        >>> footer_factor = 0.9
        >>> font_size = 8.0
        >>> remove_header_footer(blocks, page_height, footer_factor, font_size)
    """
    categorizer_vectors = [(ele, (i,j)) for i, b in enumerate(blocks) for j, ele in enumerate(b)]
    labels = text_block_categorizer(categorizer_vectors, min_sample, eps)

    # 0 = normal text | 1 = header and footer
    for l, e in zip(labels, categorizer_vectors):
        ele = e[0]
        if (l == 1 or re.fullmatch(r'\d', ele['text']) or ele['element'].bbox[1] < page_height * footer_factor or ele['font_size'] < font_size)\
            and not re.match(r'\b(edital|chamada|seleção|programa)\b', ele['text'], re.IGNORECASE | re.UNICODE):
            i, j = e[-1]
            blocks[i][j]['text'] = ''

def extract_numbered_sections(blocks, normal_font_size, text_spacing, dist_factor_same_block=2.5, max_title_words=12):
    """
    Extract numbered sections from the document blocks list and categorize them into separate sections.

    Args:
        blocks (list): A list of text blocks, where each block is represented as a list of dictionaries containing text, font weight, font size, etc.
        normal_font_size (float): The normal font size threshold used to differentiate between main titles and presentation text.
        text_spacing (float): The average spacing between lines of text.
        dist_factor_same_block (float, optional): The distance factor threshold used to determine if a line of text belongs to the same block as the previous line. Defaults to 2.5.
        max_title_words (int, optional): The maximum number of words allowed in a main title. Defaults to 12.

    Returns:
        dict: A dictionary containing the extracted numbered sections with their titles as keys and contents as values.

    Note:
        This function extracts numbered sections from the document blocks list and categorizes them into separate sections.
        It identifies main titles based on bold font weight and presentation text based on font size. It then associates presentation text
        with the nearest main title and collects them into sections.

    Example:
        >>> blocks = [
        ...     [{'text': '1. Introduction', 'font_weight': 'bold', 'font_size': 14.0, ...}, {'text': 'Introduction content', 'font_weight': 'normal', 'font_size': 12.0, ...}, ...],
        ...     [{'text': '2. Methodology', 'font_weight': 'bold', 'font_size': 14.0, ...}, {'text': 'Methodology content', 'font_weight': 'normal', 'font_size': 12.0, ...}, ...],
        ...     ...
        ... ]
        >>> normal_font_size = 12.0
        >>> text_spacing = 20.0
        >>> dist_factor_same_block = 2.5
        >>> max_title_words = 12
        >>> sections = extract_numbered_sections(blocks, normal_font_size, text_spacing, dist_factor_same_block, max_title_words)
    """
    sections = {}
    headers = []
    current_title = None
    # looking for numbered sections
    for i, b in enumerate(blocks):
        if is_title_section(b[0], max_words=max_title_words):
            if current_title:
                sections[current_title] = current_section
            current_title = b[0]['text']
            current_section = []

            # check if inside the block of the title has some text into it
            if len(b) > 1:
                texts = ' '.join([line['text'] for line in b[1:]])
                current_section.append(texts)

        elif current_title:
            texts = ' '.join([line['text'] for line in b])
            current_section.append(texts)

        else:
            headers.append(b)
    
    # Save the last section
    if current_title:
        sections[current_title] = current_section
        current_title = None

    # if has some text in the sections dictionary means that the content inside the headers is part of the header (main title + presentation)
    # looking for main title and presentaion
    if len(sections) > 0:
        main_title = []
        text_presentation = []
        for i in headers:
            block_text = []
            for l in i:
                # Generaly the title comes with bold font weight
                if l['font_weight'] == 'bold':
                    if len(main_title) > 0:
                        prev_line = main_title[-1]
                        dist_line = calculate_distance(prev_line['element'], l['element'])
                        # calculating if the text block is part of the prev text block
                        if (dist_line < (dist_factor_same_block*text_spacing) or re.match(r'\b(edital|chamada|seleção|programa)\b', l['text'], re.IGNORECASE | re.UNICODE)) and len(l['text'].strip().split()) <= max_title_words:
                            main_title.append(l)
                        else:
                            text_presentation.append(l['text'])
                    else:
                        main_title.append(l)
                # if it's not a title so it's a presentation text
                elif l['font_size'] >= normal_font_size:
                    block_text.append(l['text'])
            
            text_presentation.append(' '.join(block_text))

        main_title = ' '.join(t['text'].strip() for t in main_title)
        sections = {main_title:text_presentation, **sections}

    return sections

def extract_text_propriety_sections(blocks, font_sizes, max_blocks_main_title, text_spacing, dist_factor_same_block=4):
    """
    Extract sections based on text propriety such as font size, font weight, and page number.

    Args:
        blocks (list): A list of text blocks, where each block is represented as a list of dictionaries containing text, font weight, font size, etc.
        font_sizes (list): A list of font sizes corresponding to the text blocks.
        max_blocks_main_title (int): The maximum number of blocks considered as a main title.
        text_spacing (float): The average spacing between lines of text.
        dist_factor_same_block (float, optional): The distance factor threshold used to determine if a line of text belongs to the same block as the previous line. Defaults to 4.

    Returns:
        dict: A dictionary containing the extracted sections with their titles as keys and contents as values.

    Note:
        This function extracts sections based on text propriety such as font size, font weight, and page number.
        It identifies main titles based on font size and font weight and associates subsequent text blocks with the nearest main title.

    Example:
        >>> blocks = [
        ...     [{'text': 'Introduction', 'font_weight': 'bold', 'font_size': 14.0, 'page': 0, ...}, {'text': 'Introduction content', 'font_weight': 'normal', 'font_size': 12.0, 'page': 0, ...}, ...],
        ...     [{'text': 'Methodology', 'font_weight': 'bold', 'font_size': 14.0, 'page': 0, ...}, {'text': 'Methodology content', 'font_weight': 'normal', 'font_size': 12.0, 'page': 0, ...}, ...],
        ...     ...
        ... ]
        >>> font_sizes = [14.0, 12.0, ...]
        >>> max_blocks_main_title = 2
        >>> text_spacing = 20.0
        >>> dist_factor_same_block = 4
        >>> sections = extract_text_propriety_sections(blocks, font_sizes, max_blocks_main_title, text_spacing, dist_factor_same_block)
    """
    sections = {}
    current_title = None
    prev_line = None
    title_font_size = max([size for size, freq in Counter(font_sizes).most_common() if freq > 10])
    for i, b in enumerate(blocks):
        if is_title_text_propriety(b[0], title_font_size) or b[0]['page'] == 0 and b[0]['font_size'] > title_font_size:
            # trying to find the main title and presentarion at first blocks of the document and checking 
            # if the current text block is part of the prev text block
            if prev_line and current_title and i < max_blocks_main_title:
                dist_line = calculate_distance(prev_line['element'], b[0]['element'])
                if dist_line < (dist_factor_same_block*text_spacing):
                    prev_line = b[0]
                    current_title += ' '+b[0]['text']
                else:
                    prev_line = b[0]
                    current_title = b[0]['text']
                continue

            elif current_title:
                sections[current_title] = current_section

            prev_line = b[0]
            current_title = b[0]['text']
            current_section = []

        elif current_title:
            texts = ' '.join([line['text'] for line in b])
            current_section.append(texts)

    # Save the last section
    if current_title:
        sections[current_title] = current_section
        current_title = None

    return sections

def extract_content(pdf_path, char_margin=40, line_margin=0.1, boxes_flow=-1, text_spacing_min_freq=50, max_blocks_space_search_main_title=4):
    """
    Extract content from a PDF file by analyzing various properties of the text elements.

    Args:
        pdf_path (str): The path to the PDF file.
        char_margin (int, optional): The margin for characters. Defaults to 40.
        line_margin (float, optional): The margin between lines. Defaults to 0.1.
        boxes_flow (int, optional): The flow of text boxes. Defaults to -1.
        text_spacing_min_freq (int, optional): The minimum frequency of text spacing values. Defaults to 50.
        max_blocks_space_search_main_title (int, optional): The maximum number of blocks to search for main titles based on spacing. Defaults to 4.

    Returns:
        dict: A dictionary containing the extracted content with titles as keys and content sections as values.

    Note:
        This function extracts content from a PDF file by analyzing various properties of the text elements, such as font size, font weight, spacing, etc.
        It first attempts to extract numbered sections. If no numbered sections are found, it analyzes text propriety (font weight, font size, and text bounding box) to identify sections.

    Example:
        >>> pdf_path = 'example.pdf'
        >>> char_margin = 40
        >>> line_margin = 0.1
        >>> boxes_flow = -1
        >>> text_spacing_min_freq = 50
        >>> max_blocks_space_search_main_title = 4
        >>> sections = extract_content(pdf_path, char_margin, line_margin, boxes_flow, text_spacing_min_freq, max_blocks_space_search_main_title)
    """
    pages = list(extract_pages(pdf_path, laparams=LAParams(char_margin=char_margin, line_margin=line_margin, boxes_flow=boxes_flow)))
    font_sizes, text_spaces, page_height = get_pages_properties(pages)
    text_spacing_list = [space for space, freq in Counter(text_spaces).most_common() if freq > text_spacing_min_freq and space > 0]
    normal_font_size = mode(font_sizes)
    blocks = extract_text_blocks(pages, pdf_path, mode(text_spaces))
    remove_header_footer(blocks, page_height, 0.028, normal_font_size)

    if len(text_spacing_list) > 0:
        text_spacing = min(text_spacing_list)
    else:
        text_spacing = mode(text_spaces)

    sections = extract_numbered_sections(blocks, normal_font_size, text_spacing)
    # no numbered sections found
    # analyzing by font propriety (font weight, font size and text bounding box)
    if len(sections) == 0:
        sections = extract_text_propriety_sections(blocks, font_sizes, max_blocks_space_search_main_title, text_spacing)
    return sections