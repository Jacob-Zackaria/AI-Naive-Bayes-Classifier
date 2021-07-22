#ifndef FILES_ACCESS_H
#define FILES_ACCESS_H

#include <fstream>
#include <vector>
#include <string>
#include "../Category.h"

// structure representing a document and it's category.
struct DocumentCategory
{
	unsigned int documentIndex;
	Category category;
};

// function which gets documents and their corresponding categories.
void GetDocumentCategories(const char* fileName, std::vector<DocumentCategory>& newDocumentCategory);

// function to get terms as strings from file.
void GetTerms(const char* fileName, std::vector<std::string>& newTerms);

// function to get matrix of each terms from file.
void GetMatrix(const char* fileName, std::vector<std::vector<float>>& newMatrix);


#endif FILES_ACCESS_H