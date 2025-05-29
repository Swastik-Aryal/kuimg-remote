import csv
import pdfplumber
import nltk
from nltk.corpus import stopwords
import string
import yake


# RUN THIS IF RUNNING LOCALLY AND NOT USING DOCKER otherwise taken care of in Dockerfile

# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)


# Removes punctuation
# Keep only alphabetic words and join into a single string
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text into words and remove stop words
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    processed_text = " ".join(
        [word for word in tokens if word not in stop_words and word.isalpha()]
    )
    return processed_text


# extract keywords
def keyword_extractor(
    text,
    language="en",
    max_ngram_size=1,
    deduplication_threshold=0.9,
    deduplication_algo="seqm",
    windowSize=1,
    numofextractions=20,
):
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=windowSize,
        top=numofextractions,
        features=None,
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords


def filter_keywords(keywords, file_path, numofkeywords=10):
    # Filter candidates based on concreteness scores
    concreteness_dict = {}
    keywords_candidates = [
        kw[0] for kw in keywords if kw[0] != "percent"
    ]  # Extract keyword strings from YAKE results

    # Open the file and process concreteness scores
    with open(file_path, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        keywords_set = set(
            keywords_candidates
        )  # Convert keywords to a set for faster lookup
        for row in csv_reader:
            word = row[0]
            if word in keywords_set:
                concreteness_dict[word] = float(row[2])

    # Sort keywords based on concreteness scores (higher scores first)
    sorted_keywords = sorted(
        keywords_candidates,
        key=lambda word: concreteness_dict.get(word, 0),
        reverse=True,
    )
    # Get the top N keywords based on numofkeywords
    top_keywords = sorted_keywords[:numofkeywords]

    return {word: concreteness_dict.get(word, 0) for word in top_keywords}


def extract_keywords_from_pdf(
    pdf_path,
    language="en",
    max_ngram_size=1,
    deduplication_threshold=0.9,
    deduplication_algo="seqm",
    windowSize=1,
    numOfKeywords=10,
    file_path="Pdf_training/concreteness_scores_original.csv",
    as_csv=False,
):
    """
    Extract keywords from a PDF file using YAKE algorithm and filter them based on concreteness scores.
    This function extracts text from a PDF file, preprocesses it, extracts keywords using YAKE,
    and then filters these keywords based on concreteness scores from an external file.
    Parameters:
        pdf_path (str): Path to the PDF file to extract keywords from
        language (str, optional): Language for YAKE. Defaults to "en".
        max_ngram_size (int, optional): Maximum n-gram size for keyword extraction. Defaults to 1.
        deduplication_threshold (float, optional): Threshold for deduplication in YAKE. Defaults to 0.9.
        deduplication_algo (str, optional): Algorithm for deduplication in YAKE. Defaults to "seqm".
        windowSize (int, optional): Window size for YAKE. Defaults to 1.
        numOfKeywords (int, optional): Number of keywords to extract after filtering. Defaults to 20.
        file_path (str, optional): Path to the concreteness scores file. Defaults to "concreteness_scores_original.csv".
        as_csv (bool, optional): If True, saves filtered keywords to "filtered_keywords.csv" file. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - list: Filtered keywords
            - dict: Corresponding scores for the filtered keywords

    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        # Extract text from each page of the PDF
        for page in pdf.pages:
            text += page.extract_text()

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Extract keywords using YAKE
    keywords = keyword_extractor(
        processed_text,
        language,
        max_ngram_size,
        deduplication_threshold,
        deduplication_algo,
        windowSize,
        numOfKeywords * 2,
    )

    # Filter keywords based on concreteness scores
    filtered_keywords = filter_keywords(keywords, file_path, numOfKeywords)

    if as_csv:
        # Write filtered keywords to a CSV file
        with open("filtered_keywords.csv", mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            for keyword, score in filtered_keywords.items():
                csv_writer.writerow([keyword, score])
        print("Filtered keywords written to filtered_keywords.csv")

    return filtered_keywords


if __name__ == "__main__":
    # Example usage
    pdf_path = "chap1.pdf"  # Path to your PDF file
    keywords = extract_keywords_from_pdf(
        pdf_path,
        language="en",
        max_ngram_size=1,
        deduplication_threshold=0.9,
        deduplication_algo="seqm",
        windowSize=1,
        numOfKeywords=10,
        file_path="concreteness_scores_original.csv",
        as_csv=True,
    )
    print("Extracted Keywords:", keywords)
