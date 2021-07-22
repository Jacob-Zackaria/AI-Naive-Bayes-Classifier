#include "FilesAccess.h"

// function which gets documents and their corresponding categories.
void GetDocumentCategories(const char* fileName, std::vector<DocumentCategory>& newDocumentCategory)
{
	// temporary structure to read from file.
	DocumentCategory tempCategory = {};

	// open file to read.
	std::ifstream myFile(fileName);

	// string to store from file.
	char* newString = new char[100];

	// read until file is empty.
	while (!myFile.eof())
	{
		// read document index from file.
		myFile.getline(newString, 10, '\t');

		if (*newString != '\0')
		{
			// if string is not null
			if (newString != nullptr)
			{
				// store document index in temporary structure.
				sscanf_s(newString, "%u", &tempCategory.documentIndex);
			}

			// read class category from file. { delimitting character '\n' by default }
			myFile.getline(newString, 10);

			// if string is not null
			if (newString != nullptr)
			{
				// temporary variable.
				unsigned int tempVariable = 0;

				// convert string value to unsigned integer.
				sscanf_s(newString, "%u", &tempVariable);

				// store class category in temporary structure.
				tempCategory.category = (Category)tempVariable;
			}

			// store temporary category values to new vector.
			newDocumentCategory.push_back(tempCategory);
		}
	}

	// delete storage.
	delete[] newString;

	// close file.
	myFile.close();
}

// function to get terms as strings from file.
void GetTerms(const char* fileName, std::vector<std::string>& newTerms)
{
	// open file to read.
	std::ifstream myFile(fileName);

	// string to store from file.
	char* newString = new char[100];

	// read until file is empty.
	while (!myFile.eof())
	{
		// read term from file. { delimitting character '\n' by default }
		myFile.getline(newString, 100);
		
		// if string is not null
		if (newString != nullptr && *newString != '\0')
		{
			// store string to vector.
			newTerms.push_back(newString);
		}	
	}

	// delete storage.
	delete[] newString;

	// close file.
	myFile.close();
}

// function to get matrix of each terms from file.
void GetMatrix(const char* fileName, std::vector<std::vector<float>>& newMatrix)
{
	// open file to read.
	std::ifstream myFile(fileName);

	// string to store from file.
	char* newString = new char[10000];

	// temporary vector to store floats.
	std::vector<float> tempFloats = {};

	// read until file is empty.
	while (!myFile.eof())
	{
		// clear temporary vector
		tempFloats.clear();

		// read line from file.
		myFile.getline(newString, 10000, '\n');

		if (*newString != '\0')
		{
			// store in process string.
			std::string processString(newString);

			// find '\t' character.
			size_t pos = processString.find('\t');

			// iterate until process string is empty.
			while (pos != -1)
			{
				// store value temporary float.
				tempFloats.push_back(std::stof(processString.substr(0, pos)));

				// truncate the string.
				processString = processString.substr(pos + 1);

				// find next '\t' character.
				pos = processString.find('\t');

				if (pos == -1)
				{
					// store value temporary float.
					tempFloats.push_back(std::stof(processString));
				}
			}

			// store temporary vector to new matrix vector.
			newMatrix.push_back(tempFloats);
		}
	}

	// delete storage.
	delete[] newString;

	// close file.
	myFile.close();
}