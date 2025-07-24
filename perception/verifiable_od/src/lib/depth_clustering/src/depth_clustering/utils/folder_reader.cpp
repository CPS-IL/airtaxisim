// Copyright (C) 2017  I. Bogoslavskyi, C. Stachniss, University of Bonn

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "./folder_reader.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace depth_clustering
{

namespace fs = boost::filesystem;
using std::string;
using std::vector;
using boost::algorithm::ends_with;
using boost::algorithm::starts_with;

int
num_from_string(const std::string& query_str)
{
	if (query_str.empty())
	{
		return 0;
	}

	// Extract file name from path
	std::stringstream ss(query_str);
	std::string substring = "";
	while (std::getline(ss, substring, '/'))
	{
	}
	const std::string &substring_const = substring;

	const char *pattern = "\\d+";
	boost::regex re(pattern);
	boost::match_results<std::string::const_iterator> what;
	auto start = substring_const.begin();
	auto end = substring_const.end();
	int found_num = 0;
	while (boost::regex_search(start, end, what, re))
	{
		start = what[0].second;
		found_num = std::stoi(what[0].str());
	}
	return found_num;
}

bool
numeric_string_compare(const std::string& s1, const std::string& s2)
{
	return num_from_string(s1) < num_from_string(s2);
}

FolderReader::FolderReader(const string& folder_path, const string& ending_with, const Order order) :
		FolderReader(folder_path, "", ending_with, order)
{
}

FolderReader::FolderReader(const string& folder_path, const string& starting_with,
		const string& ending_with, const Order order) :
		_path_counter(0)
{
	fs::path folder(folder_path);
	if (!fs::exists(folder))
	{
		std::cout << "[WARN]: Invalid folder path \"" << folder_path.c_str() << "\"." << std::endl;
		return;
	}
	if (fs::is_directory(folder))
	{
		std::cout << "[INFO]: Opened folder \"" << folder_path.c_str() << "\"." << std::endl;

		auto range = boost::iterator_range<fs::directory_iterator>(fs::directory_iterator(folder),
				fs::directory_iterator());
		for (auto &entry : range)
		{
			auto filename = entry.path().filename().string();
			bool is_of_correct_type = starts_with(filename, starting_with)
					&& ends_with(filename, ending_with);
			if (is_of_correct_type)
			{
				_all_paths.push_back(entry.path().string());
			}
		}
		if (order == Order::SORTED)
		{
			std::sort(_all_paths.begin(), _all_paths.end(), numeric_string_compare);
		}

		std::cout << "[INFO]: Read " << _all_paths.size() << " \"" << ending_with.c_str()
				<< "\" files from folder \"" << folder_path.c_str() << "\"." << std::endl;
	}
}

string
FolderReader::GetNextFilePath()
{
	if (_path_counter < _all_paths.size())
	{
		return _all_paths[_path_counter++];
	}
	std::cout << "[INFO]: Reached last file path." << std::endl;
	return "";
}

}  // namespace depth_clustering
