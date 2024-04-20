# Importação das bibliotecas a serem utilizadas
from sklearn.pipeline import Pipeline
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# Leitura do dataframe e verificação das colunas dentro da base de dados
df = pd.read_csv('Tweets_Mg.csv')
df.head()
df.columns
# Contar a quantidade de linhas de Tweets NEUTROS, POSITIVOS e NEGATIVOS
df.Classificacao.value_counts()

# Pré-Processamento dos DADOS
# Remover linhas duplicadas na base de dados - Coluna TEXT
# Remove duplicados, inplace = True, significa que alteraremos o dataframe
df.drop_duplicates(['Text'], inplace=True)
df.Text.count()  # Linhas que sobraram

# Separando os tweets e suas classificações em variaveis separadas
tweets = df['Text']
classes = df['Classificacao']

# Instalando biblioteca e baixando bases de dados da NLTK
# Palavras que se repetem e sem "importâcia" ex: (a, e, ao, em... etc)
nltk.download('stopwords')
nltk.download('rslp')  # Removedo de sufixos da lingua Portuguesa
# Tokenização: Processo de reduzir um texto em unidades menores como palavras ou frase
nltk.download('punkt')
# Banco de dados lexical que agrupa palavras em conjuntos de sinonimo
nltk.download('wordnet')

# Funções de Pré-Processamento de dados


def RemoveStopWords(instancia):
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))
# stopwrods = Carregamos uma lista de stopwords em portugues para a função RemoveStopWords.
# palavras = Passamos uma frase pra essa função, ela interara sobre cada palavra da frase e verificara se ela é
# ou não uma stopwrod. Se for, a palavra será retirada, se não, é mantida.


def Stemming(instancia):
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for w in instancia.split():
        palavras.append(stemmer.stem(w))
    return (" ".join(palavras))
# Aplica o método stemmer, ou seja, reduz a palavra ao seu radical diminuindo o vocabulario do dataset


def Limpeza_dados(instancia):
    instancia = re.sub(r"http\S+", "", instancia).lower().replace(',',
                                                                  '').replace('.', '').replace(';', '').replace('-', '').replace(':', '')
    return (instancia)
# Essa limpeza de dados remove links, pontuação entre outros caracteres.


wordnet_lemmatizer = WordNetLemmatizer()


def Lemmatization(instancia):
    palavras = []
    for w in instancia.split():
        palavras.append(wordnet_lemmatizer.lemmatize(w))
    return (" ".join(palavras))
# Processo de lemmatização: Reduzir uma palavra a sua forma base (canonica), infinitivo para verbos e singular
# para adjetivos e substantivos.


# Entendendo como funcionam as funções
# 1 RemoveStopWords:
RemoveStopWords('Eu adoro esse tipo de conteúdo, gostaria de ver mais')
# 2 Steeming:
Stemming('Eu adoro esse tipo de conteúdo, gostaria de ver mais')
# 3 Limpeza de dados:
Limpeza_dados('Eu adoro esse tipo de conteúdo, gostaria de ver mais')
# 4 Lemmatization:
Lemmatization('Eu adoro esses tipos de conteúdos, gostaria de ver mais')

# TOKENIZAÇÃO:
# Identificam as palavras com base nos espaços entre elas. Porém, um tokenizador comum separa, também, @,# e emojis.
# o que no caso específico do tweeter não é algo interessante, pois é uma rede social que se utiliza muito
# de sinais gráficos como parte da comunicação. Pra resolver esse problema usamos o TweetTokenizer

# Ex Tokenizador:
frase = 'A live do @Andy é incrível. Cara, surreal de bom :), #TheBest'
word_tokenize(frase)
# Vemos que todos os caracteres especiais também foram removidos, dessa forma, o uso das hastags, emojis e expressões
# do tweeter não é verificado.

# Tokenizador do Tweeter:
tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize(frase)
# Aqui vemos que os tokens gerados respeitam o modus operandi da rede social :)

# CRIAÇÃO DO MODELO DE MACHINE LEARNING PARA ANALISE DE SENTIMENTO

# Intancia do Objeto que faz a vetorização dos dados de texto (Transforma uma palavra em um formato numérico para
# que o computador possa identificar aquela palavra)

# Método BagOfWords = Ele vetoriza a frequência com que cada palavra aparece criando um dicionario com todas as
# palavras presente na base de dados. Depois disso, cada sentençã é analisada com base nesse dicionario e 1 é inserido
# onde a palavra aparece e 0 onde a palavra não aparece.
# Ex: Frase1 = O gato comeu arroz
# EX: Frase2 = O cachorro comeu milho
# Dicionário criado = [O, gato, cachorro, comeu, arroz, milho]
# Vetor frase 1 = [1,1,0,1,1,0]
# Vetor frase 2 = [1,0,1,1,0,1]

# vetorizador criado, passando
vectorizer = CountVectorizer(
    analyzer="word", tokenizer=tweet_tokenizer.tokenize)
# o tweet tokanizer dentro de sua função.

# Aplicar o vetorizador nos dados de texto
freq_tweets = vectorizer.fit_transform(tweets)
type(freq_tweets)
# Número de Linhas = número de linhas da base tweets, e as colunas são as palavras encontradas.
freq_tweets.shape

# Utilizaremos o algoritmo: MULTINOMIALNB
modelo = MultinomialNB()
# Passamos a nossa matriz vetorizada e a classificação de palavras como X e Y (respctivamente)
modelo.fit(freq_tweets, classes)

# Vendo a matriz:
freq_tweets.A

# TESTANDO O MODELO COM ALGUMAS INSTÂNCIAS SIMPLES
testes = ['Esse governo está no início, vamos aguardar',
          'Devemos nos preocupar mais com a política',
          'Todos os políticos são horriveis para o povo',
          'A segurança desse pais está deixando a desejar',
          'O governo de Minas é mais uma vez do PT']

# Vetorizar os testes: Transforma os dados de testes em vetores de palavras
freq_testes = vectorizer.transform(testes)

# Fazendo a classificação com o modelo treinado:
for t, c in zip(testes, modelo.predict(freq_testes)):
    print(t + ": "+c)

# Calculando a probalidade de cada classe: Positivo, Neutro e Negativo:
print(modelo.classes_)
# Dentro do parametro round passamos a qtd de casas decimais que queremos ver.
modelo.predict_proba(freq_testes).round(2)

# TAG DE NEGAÇÕES: Uma técnica de melhoria do algoritmo, inserindo uma tag de negação após cada palavra destacada como negativa:


def marque_negacao(texto):
    negacoes = ['nao', 'not', 'não']
    negacao_detectada = False
    resultado = []
    palavras = texto.split()
    for p in palavras:
        p = p.lower()
        if negacao_detectada == True:
            p = p + '_NEG'
        if p in negacoes:
            negacao_detectada = True
        resultado.append(p)
    return (" ".join(resultado))

# testar nossa função:


marque_negacao('Eu gosto pessego')
marque_negacao('Eu não gosto de pessego')

# Criando modelos usando PIPELINES
# Pipelines são interessantes para reduzir código e automatizar fluxos.

pipeline_simples = Pipeline(
    [('counts', CountVectorizer()), ('classifier', MultinomialNB())])
# Inserimos dentro do pipeline_simples a ação de vetorizar uma entrada de palavras e depois, inserir dentro do algoritmo MultinoalNB

# Pipeline que atribiu a tag de negação:
pipeline_negacoes = Pipeline([
    ('counts', CountVectorizer(tokenizer=lambda text: marque_negacao(text))),
    ('classifier', MultinomialNB())
])

# Treinar nosso modelo dentro dos pipelines:
pipeline_simples.fit(tweets, classes)
pipeline_simples.steps

# Treinar nosso modelo dentro do pipeline de negacoes:
pipeline_negacoes.fit(tweets, classes)

# VALIDANDO OS MODELOS COM VALIDAÇÃO CRUZADA PARA O PRIMEIRO PIPELINE:

resultados = cross_val_predict(pipeline_simples, tweets, classes, cv=10)

# Medindo a acuracia média do modelo:
metrics.accuracy_score(classes, resultados)

# Medidas de validação
sentimento = ['Positivo', 'Neutro', 'Negativo']
print(metrics.classification_report(classes, resultados, sentimento))

# Matriz de Confusão:
print(pd.crosstab(classes, resultados, rownames=[
      'Real'], colnames=['Preditivo'], margins=True))

# VALIDANDO OS MODELOS COM VALIDAÇÃO CRUZADA PARA O PIPELINE DE TAG DE NEGAÇÃO:

results = cross_val_predict(pipeline_negacoes, tweets, classes, cv=10)

# Meidndo a acuracia média do modelo;
metrics.accuracy_score(classes, results)

# Medidas de Validação:
sentimento = ['Positivo', 'Neutro', 'Negativo']
print(metrics.classification_report(classes, results, sentimento))

# Matriz de Confusão:
print(pd.crosstab(classes, results, rownames=[
      'Real'], colnames=['Preditivo'], margins=True))

# VISTO OS DOIS MODELOS, CONCLUIMOS QUE PARA ESSE EXEMPLOS, COLOCAR A TAG DE NEGAÇÃO, DIMINUI A EFICACIA DO MODELO.

### BOAS PRÁTICAS:

#1. Tente aumentar os features para compor o modelo, caso seja possivel.