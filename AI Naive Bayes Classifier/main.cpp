#include "NaiveBayesClassifier.h"

// program start.
int main()
{
	// Files needed.
	const char* termsFile = "DataSet/terms.txt";
	const char* trainClasses = "DataSet/TrainingSet/trainClasses.txt";
	const char* trainMatrix = "DataSet/TrainingSet/trainMatrix.txt";
	const char* testClasses = "DataSet/TestSet/testClasses.txt";
	const char* testMatrix = "DataSet/TestSet/testMatrix.txt";

	// trains model with terms file, class file and train matrix file.
	NaiveBayesClassifier::TrainModel(termsFile, trainClasses, trainMatrix);

	// vector of categorized documents.
	std::vector<DocumentCategory> categorizedDocuments = {};

	// tests model with given test matrix file { returns categorized documents }.
	NaiveBayesClassifier::TestModel(testMatrix, categorizedDocuments);

	// Evaluate accuracy of model by comparing with actual test class file.
	NaiveBayesClassifier::EvaluateModel(categorizedDocuments, testClasses);

	// outputs the class probability for a particular term file from trained data.
	NaiveBayesClassifier::GetTermClassProbabilities(termsFile);

	// success.
	return 0;
}