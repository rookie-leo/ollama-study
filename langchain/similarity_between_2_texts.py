import time
import spacy

def load_model():
    try:
        return spacy.load('en_core_web_md')
    except OSError:
        raise RuntimeError(
            "The model 'en_core_web_md' is not installed. "
            "Install it using: python -m spacy download en_core_web_md"
        )

def calculate_similarity(nlp, text1: str, text2: str) -> float:
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    return doc1.similarity(doc2)

def main():
    nlp = load_model()

    text1 = """
    The findings reveal an impressive average accuracy of 99% between the CL values obtained
    from analytical methods and the DNN model. This high level of agreement demonstrates the
    DNN model's exceptional capability in accurately predicting and designing NACA 4-digit airfoils.
    """

    text2 = """
    To further validate the DNN model's performance, we designed 10,000 airfoils using both
    analytical methods and the DNN model. The results show that the DNN model can design
    airfoils four times faster than the analytical method. This significant time advantage highlights
    the computational efficiency of the DNN approach, which is crucial in practical applications
    where rapid design iterations are necessary.
    """

    start = time.perf_counter()

    for _ in range(1000):
        score = calculate_similarity(nlp, text1, text2)

    end = time.perf_counter()

    print(f'Tempo de execução (1000 execuções): {end - start:.1f} segundos')
    print(f'Pontos de similaridade: {score:.2f}')

if __name__ == '__main__':
    main()


