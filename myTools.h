#pragma once

#include <string>
//#include <wstring>
#include <vector>

class myTools {
public:

	static bool readSourceFile(std::string & str, const std::string & filename);
	// TODO: micro optimization, omit some unused branch
	static const char * getPtxString(const char * filename, 
		const std::vector<std::string> * macro = nullptr,
		const std::string * cur_path=nullptr
	);

	// path operation
	static bool is_root_path(const char * filename);
	static std::string cat_path(const char * path, const char * file);
	static std::string to_lower(const std::string & s);

	// "D:/File/1.txt" -> "D:/File","1.txt"
	void path_split(const char * filename, char * path, char * file);

	//static int32_t createDirectory(const std::string &directoryPath);
	static void createDirectoryRecursively(const std::string &directory);
	static bool isValidFile(const char * filename);
	// flag = 0, all
	// flag = 1, directory
	// flag = 2, 
	std::vector<std::string> getFilenameInDirectory(const char * dirname, int flag = 0);

};