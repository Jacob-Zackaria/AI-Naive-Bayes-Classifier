#include "NaiveBayesClassifier.h"


NaiveBayesClassifier* NaiveBayesClassifier::getPrivateInstance()
{
	static NaiveBayesClassifier instance;
	return &instance;
}

NaiveBayesClassifier::NaiveBayesClassifier()
	:
	categoryProbability{},
	termCategoryProbability{},
	maxProbability{}
{
}

NaiveBayesClassifier::~NaiveBayesClassifier()
{
}

// get document categorized.
void NaiveBayesClassifier::GetDocByCategory(std::vector<DocumentCategory>& newDocumentCategory, Category newCategory, std::vector<DocumentCategory>& docByCategory)
{
	// loop through all document category vector.
	for (uint32_t i = 0; i < newDocumentCategory.size(); i++)
	{
		// if matches given category
		if (newDocumentCategory[i].category == newCategory)
		{
			// add to new vector.
			docByCategory.push_back(newDocumentCategory[i]);
		}
	}
}

// count term by category
const float NaiveBayesClassifier::CountTermByCategory(std::vector<std::vector<float>>& newMatrix, std::vector<DocumentCategory>& docByCategory, const uint32_t term)
{
	// term count
	float numberOfTermsInCategory = 0;

	// loop through all document category vector.
	for (uint32_t i = 0; i < docByCategory.size(); i++)
	{
		// update count of terms
		numberOfTermsInCategory += newMatrix[term][docByCategory[i].documentIndex];
	}

	// return total count.
	return (numberOfTermsInCategory);
}

// total count of terms by category
const float NaiveBayesClassifier::TotalTermByCategory(std::vector<std::vector<float>>& newMatrix, std::vector<DocumentCategory>& docByCategory)
{
	// term count
	float totalNumberTermsInCategory = 0;

	// loop through all document category vector.
	for (uint32_t i = 0; i < docByCategory.size(); i++)
	{
		// loop through all terms in matrix vector.
		for (uint32_t j = 0; j < newMatrix.size(); j++)
		{
			// update count of terms
			totalNumberTermsInCategory += newMatrix[j][docByCategory[i].documentIndex];
		}
	}

	// return total count.
	return (totalNumberTermsInCategory);
}

// calculate probability of each terms in document using conditional probabilities. { sum(log(P(ai | Cj))) where 'sum' ranges from '1' to 'category count' }
const float NaiveBayesClassifier::CalculateTermCategoryProbability(std::vector<std::vector<float>>& newMatrix, const uint32_t docIndex, Category newCategory)
{
	// get private instance.
	NaiveBayesClassifier* pClassifier = NaiveBayesClassifier::getPrivateInstance();

	// product of all term probability.
	float termProbability = 0.0f;

	// for each term.
	for (uint32_t i = 0; i < newMatrix.size(); i++)
	{
		// get number of times term occurs in document. 
		float termOccurence = newMatrix[i][docIndex];

		// assigns term category probability based on catgory.
		termProbability += (termOccurence * logf(pClassifier->termCategoryProbability[((unsigned int)newCategory * newMatrix.size()) + i]));
	}

	// return calculated probability.
	return (termProbability);
}

// calculate category based on all term category probability { max (log(P(Cj)) + sum(log((P(ai | Cj))) }
const Category NaiveBayesClassifier::FindMaxProbability(std::vector<float>& documentCategoryProbability)
{
	// get private instance.
	NaiveBayesClassifier* pClassifier = NaiveBayesClassifier::getPrivateInstance();

	// maximum of probabilty
	float newMaxProbability = documentCategoryProbability[0];

	// catgeory of maximum probability.
	Category maxCategory = {};

	// iterate all category probabilities obtained for current document.
	for (uint32_t i = 0; i < documentCategoryProbability.size(); i++)
	{
		// if current category probability is higher than previous.
		if (documentCategoryProbability[i] > newMaxProbability)
		{
			// set new max as current category probability.
			newMaxProbability = documentCategoryProbability[i];

			// set category of new max category probability.
			maxCategory = (Category)i;
		}
	}

	// set max probability
	pClassifier->maxProbability.push_back(newMaxProbability);

	// return max category.
	return (maxCategory);
}

// trains model with terms file, class file and train matrix file.
void NaiveBayesClassifier::TrainModel(const char* termsFile, const char* trainClasses, const char* trainMatrix)
{
	printf("\n Training Model :");

	// get private instance.
	NaiveBayesClassifier* pClassifier = NaiveBayesClassifier::getPrivateInstance();

	//----------------------> FILE READ <-----------------------------//
	
	// terms vector.
	std::vector<std::string> newTerms = {};

	// get terms from file. { size = number of terms }
	GetTerms(termsFile, newTerms);

	// document category vector.
	std::vector<DocumentCategory> newDocumentCategory = {};

	// get document category from file. { size = number of documents }
	GetDocumentCategories(trainClasses, newDocumentCategory);
	
	// matrix vector. { row = terms, columns = document }
	std::vector<std::vector<float>> newMatrix = {};

	// get matrix from file. { row size = number of terms, column size = number of documents }
	GetMatrix(trainMatrix, newMatrix);

	//----------------------------------------------------------------//

	//----------------------> INITIALIZATION <------------------------//

	// reserve by category count * count of terms { for performance }
	pClassifier->termCategoryProbability.reserve((unsigned int)Category::CATEGORY_COUNT * newTerms.size());

	// vector to store categorized documents.
	std::vector<DocumentCategory> docByCategory = {};

	//----------------------------------------------------------------//

	//----------------------> ITERATION <-----------------------------//
	
	// for each category
	for (unsigned int i = 0; i < (unsigned int)Category::CATEGORY_COUNT; i++)
	{
		// clear category vector.
		docByCategory.clear();

		// get documents categorized. { Ti }
		pClassifier->GetDocByCategory(newDocumentCategory, (Category)i, docByCategory);

		// probability of each category. { P(Ci) = | Ti | / | D | }
		pClassifier->categoryProbability[i] = (float)docByCategory.size() / (float)newDocumentCategory.size();

		// total number of terms belonging to a particular category of documents. { ni }
		const float totalNumberTermsInCategory = pClassifier->TotalTermByCategory(newMatrix, docByCategory);

		// for each term { Wj }
		for (uint32_t j = 0; j < newTerms.size(); j++)
		{
			// number of given term belonging to a particular category of documents { nij }
			const float numberOfTermsInCategory = pClassifier->CountTermByCategory(newMatrix, docByCategory, j);

			// get conditional probability of each term. { P(Wj | Ci) = (nij + 1) / (ni + |V|) }
			const float conditionalProbability = (numberOfTermsInCategory + 1.0f) / (totalNumberTermsInCategory + newTerms.size());

			// store it in term category probability vector.
			pClassifier->termCategoryProbability.push_back(conditionalProbability);
		}
	}

	//----------------------------------------------------------------//
}

// tests model with given test matrix file { returns categorized documents }.
void NaiveBayesClassifier::TestModel(const char* testMatrix, std::vector<DocumentCategory>& docByCategory)
{
	printf("\n Testing Model :");

	// get private instance.
	NaiveBayesClassifier* pClassifier = NaiveBayesClassifier::getPrivateInstance();

	//----------------------> FILE READ <-----------------------------//
	
	// vector to store test matrix. { row = terms, columns = document }
	std::vector<std::vector<float>> newMatrix = {};

	// get matrix from file. { row size = number of terms, column size = number of documents }
	GetMatrix(testMatrix, newMatrix);

	//----------------------------------------------------------------//

	//----------------------> INITIALIZATION <------------------------//

	// temporary document category.
	DocumentCategory temporaryCategory = {};

	// reserve max probabilty.
	pClassifier->maxProbability.reserve(newMatrix[0].size());

	// probability of current document belonging to each category.
	std::vector<float> documentCategoryProbability = {};

	//----------------------------------------------------------------//

	//----------------------> ITERATION <-----------------------------//

	// for each document.
	for (uint32_t i = 0; i < newMatrix[0].size(); i++)
	{
		// add document to temporary category.
		temporaryCategory.documentIndex = i;

		// clear vector.
		documentCategoryProbability.clear();

		// for each category
		for (unsigned int j = 0; j < (unsigned int)Category::CATEGORY_COUNT; j++)
		{
			// probability of current document belonging to category { log(P(Cj)) + sum(log(P(ai | Cj))) where 'sum' ranges from '1' to 'category count' }
			documentCategoryProbability.push_back(logf(pClassifier->categoryProbability[j]) + pClassifier->CalculateTermCategoryProbability(newMatrix, i, (Category)j));
		}

		// classify document based on maximium probability of each category. { max (log(P(Cj)) + sum(log((P(ai | Cj))) }
		temporaryCategory.category = pClassifier->FindMaxProbability(documentCategoryProbability);

		// store in categorized document vector.
		docByCategory.push_back(temporaryCategory);
	}

	//----------------------------------------------------------------//
}

// Evaluate accuracy of model by comparing with actual test class file.
void NaiveBayesClassifier::EvaluateModel(std::vector<DocumentCategory>& docByCategory, const char* testClasses)
{
	printf("\n Evaluation Result :");

	// get private instance.
	NaiveBayesClassifier* pClassifier = NaiveBayesClassifier::getPrivateInstance();

	//----------------------> FILE READ <-----------------------------//

	// vector to store categorized documents.
	std::vector<DocumentCategory> testCategory = {};

	// get document category from file. { size = number of documents }
	GetDocumentCategories(testClasses, testCategory);

	//----------------------------------------------------------------//

	//----------------------> INITIALIZATION <------------------------//

	// confusion matrix.
	float confusionMatrix[2][2] = {};

	//----------------------------------------------------------------//

	//----------------------> ITERATION <-----------------------------//

	// for each document.
	for (uint32_t i = 0; i < docByCategory.size(); i++)
	{
		printf("\n Test Item: %u - ", i);
		printf("Predicted Class: %u ", (unsigned int)docByCategory[i].category);
		printf("Actual Class: %u ", (unsigned int)testCategory[i].category);
		printf("Probability: %f", pClassifier->maxProbability[i]);
	}

	// for each document.
	for (uint32_t i = 0; i < docByCategory.size(); i++)
	{
		if ((testCategory[i].category == Category::MICROSOFT_WINDOWS) && (docByCategory[i].category == Category::MICROSOFT_WINDOWS))
		{
			// True Positive (TP)
			confusionMatrix[0][0]++;
		}
		else if ((testCategory[i].category == Category::MICROSOFT_WINDOWS) && (docByCategory[i].category == Category::HOCKEY))
		{
			// False Negative (FN)
			confusionMatrix[0][1]++;
		}
		else if ((testCategory[i].category == Category::HOCKEY) && (docByCategory[i].category == Category::MICROSOFT_WINDOWS))
		{
			// False Positive (FP)
			confusionMatrix[1][0]++;
		}
		else if ((testCategory[i].category == Category::HOCKEY) && (docByCategory[i].category == Category::HOCKEY))
		{
			// True Negative (TN)
			confusionMatrix[1][1]++;
		}
	}

	// total value of confusion matrix.
	float total = (confusionMatrix[0][0] + confusionMatrix[0][1]) + (confusionMatrix[1][0] + confusionMatrix[1][1]);

	// accuracy of model. { (TP + TN) / Total }
	float accuracy = (confusionMatrix[0][0] + confusionMatrix[1][1]) / total;

	// print accuracy.
	printf("\n\nOverall Accuracy: %f\n", accuracy);

	//----------------------------------------------------------------//
}

// outputs the class probability for a particular term file from trained data.
void NaiveBayesClassifier::GetTermClassProbabilities(const char* termsFile)
{
	printf("\n\n Term Class Probabilities (Press '0' to quit):");

	// get private instance.
	NaiveBayesClassifier* pClassifier = NaiveBayesClassifier::getPrivateInstance();

	//----------------------> FILE READ <-----------------------------//

	// terms vector.
	std::vector<std::string> newTerms = {};

	// get terms from file. { size = number of terms }
	GetTerms(termsFile, newTerms);

	//----------------------------------------------------------------//

	// input string variable.
	char inputString[100];

	// iterate until user presses 'ESC' key
	while (true)
	{
		printf("\n\n Enter term :");

		// read input. { scanf stops at first whitespace character }
		RETURN_IGNORE(scanf_s("%s", inputString, 100));

		// check if user pressed '0' key
		if (*inputString == '0')
		{
			break;
		}
		else
		{
			uint32_t i = 0;

			// iterate the terms and check if the given term exist.
			for ( ; i < newTerms.size(); i++)
			{
				// if the term exist.
				if (newTerms[i].compare(inputString) == 0)
				{
					// print category 0 probability.
					printf("\n Category: Microsoft Windows   Class Probability:%f", pClassifier->termCategoryProbability[i]);

					// print category 1 probability.
					printf("\n Category: Hockey   Class Probability:%f", pClassifier->termCategoryProbability[newTerms.size() + i]);

					// break from term iteration.
					break;
				}
			}

			// if iteration completed.
			if (i == newTerms.size())
			{
				printf("\n Term doesn't exist, Try agian! or press '0' to quit!");
			}
		}
	}
}
