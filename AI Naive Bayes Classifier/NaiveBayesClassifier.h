#ifndef NAIVE_BAYES_CLASSIFIER_H
#define NAIVE_BAYES_CLASSIFIER_H

#include "FilesAccess.h"

#define RETURN_IGNORE(x) (void(x))

class NaiveBayesClassifier
{
public:

	//------------------------> Big four operators <-------------------------//

	// copy constructor.
	NaiveBayesClassifier(const NaiveBayesClassifier&) = delete;

	// copy assignment operator. 
	const NaiveBayesClassifier& operator = (const NaiveBayesClassifier&) = delete;

	// destructor.
	~NaiveBayesClassifier();

	//----------------------------------------------------------------------//

	//------------------------> Public functions <--------------------------//

	// trains model with terms file, class file and train matrix file.
	static void TrainModel(const char* termsFile, const char* trainClasses, const char* trainMatrix);

	// tests model with given test matrix file { returns categorized documents }.
	static void TestModel(const char* testMatrix, std::vector<DocumentCategory>& docByCategory);

	// Evaluate accuracy of model by comparing with actual test class file.
	static void EvaluateModel(std::vector<DocumentCategory>& docByCategory, const char* testClasses);

	// outputs the class probability for a particular term file from trained data.
	static void GetTermClassProbabilities(const char* termsFile);

	//----------------------------------------------------------------------//

private:

	//------------------------> Private functions <-------------------------//

	// get singleton instance.
	static NaiveBayesClassifier* getPrivateInstance();

	// private default constructor.
	NaiveBayesClassifier();

	// get document categorized.
	void GetDocByCategory(std::vector<DocumentCategory>& newDocumentCategory, Category newCategory, std::vector<DocumentCategory>& docByCategory);

	// count term by category
	const float CountTermByCategory(std::vector<std::vector<float>>& newMatrix, std::vector<DocumentCategory>& docByCategory, const uint32_t term);

	// total count of terms by category
	const float TotalTermByCategory(std::vector<std::vector<float>>& newMatrix, std::vector<DocumentCategory>& docByCategory);

	// calculate probability of each terms in document using conditional probabilities. { sum(log(P(ai | Cj))) where 'sum' ranges from '1' to 'category count' }
	const float CalculateTermCategoryProbability(std::vector<std::vector<float>>& newMatrix, const uint32_t docIndex, Category newCategory);

	// calculate category based on all term category probability { max (log(P(Cj)) + sum(log((P(ai | Cj))) }
	const Category FindMaxProbability(std::vector<float>& documentCategoryProbability);

	//----------------------------------------------------------------------//

	//------------------------------->  Data <------------------------------//

	// probability of each category. { P(Ci) priors }
	float categoryProbability[(unsigned int)Category::CATEGORY_COUNT];

	// probability of each term in a given category. { P(Wj | Ci) conditional probabilities }
	std::vector<float> termCategoryProbability;

	// max probability of each document. { max (log(P(Cj)) + sum(log((P(ai | Cj))) }
	std::vector<float> maxProbability;

	//----------------------------------------------------------------------//

};


#endif NAIVE_BAYES_CLASSIFIER_H