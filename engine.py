import json, string
import nltk
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer


def receiveJson():
	"""
	This function is a dummy for the final get and post requests, ignore as necessary, simply receives JSONs
	from file that correspond to the job descriptions and the data and returns them as lists of python dictionaries

	Receives: Nothing
	Outputs: a DICTIONARY AND A LIST OF DICTIONARIES
	"""
	json1_file = open('jobs.json', encoding = 'Latin-1')
	json2_file = open('data.json', encoding = 'Latin-1')
	json1_str = json1_file.read()
	json2_str = json2_file.read()
	jobs = json.loads(json1_str)
	data = json.loads(json2_str)
	return jobs, data
 	
def createResumeBody(resumeJSON):
	"""
	Parses the Resume JSON and returns the body of text that will be used to determine similarity to job descriptions

	Receives: resume JSON as a dictionary
	Outputs": A STRING CONTAINING ALL RELEVANT INFORMATION
	"""

	edList = resumeJSON['education_and_training']
	bodyText = ''
	bodyText += resumeJSON['summary'][0]["Summary"]
	bodyText += edList[0]["Certifications"] + edList[1]["Courses"] + edList[2]["Education"]
	bodyText += resumeJSON['skills'][0]['Skills & Expertise'] + resumeJSON['skills'][1]['Languages'] \
				+ resumeJSON['skills'][2]['Programming Languages']

	for jobs in resumeJSON['work_experience']:
		if 'jobtitle' in jobs:
			bodyText += jobs['jobtitle']
			bodyText += jobs['text']
		else: 
			bodyText += jobs["Projects"]
	return bodyText

def cleanAndTokenize(body):
	"""
	Cleans and tokenizes any body of text given to it (given it is a contiguous string)

	Receives: a body of text
	Returns: A tokenized, stripped, and stemmed list of tokens that will be input into the model
	"""
	stemmer = nltk.stem.porter.PorterStemmer();
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	wordList = nltk.word_tokenize(body.lower().translate(remove_punctuation_map))
	stopList = set('for a of the and to in'.split())
	texts = [word for word in wordList if word not in stopList]
	wordList = [stemmer.stem(item) for item in wordList]
	return wordList


def main():
	"""
	Executes the main functions of the prediction model, including a Latent Semantic Indexing model

	Receives: None
	Returns: A ranked JSON of job opportunities by applicant fit

	"""
	jobs, data = receiveJson()

	dataBod = createResumeBody(data)
	dataTokens = cleanAndTokenize(dataBod)
	jobsList = [cleanAndTokenize(job['text']) for job in jobs]

	dictionary = corpora.Dictionary(jobsList)
	corpus = [dictionary.doc2bow(jobs) for jobs in jobsList]
	tfidf = models.TfidfModel(corpus)
	corpus = tfidf[corpus]
	lsi = models.LsiModel(corpus, id2word=dictionary, num_topics = 1000 )
	dataVector = tfidf[dictionary.doc2bow(dataTokens)]
	index = similarities.MatrixSimilarity(lsi[corpus])
	sims = index[lsi[dataVector]]
	sims = sorted(enumerate(sims), key = lambda item: -item[1])

	print(sims)
	with open('sims.txt', 'w') as outfile:
		outfile.write(str(sims))

	retList = []
	for i in sims:
		retList.append(jobs[i[0]])
	with open('recs.json', 'w') as outfile:
		json.dump(retList, outfile)

	print(retList)
	return retList





if __name__ == "__main__":
	main()
