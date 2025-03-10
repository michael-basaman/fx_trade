/*
Copyright 2013 Michael O'Keeffe (a.k.a. ninety47).

This file is part of ninety47 Dukascopy toolbox.

The "ninety47 Dukascopy toolbox" is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or any later version.

"ninety47 Dukascopy toolbox" is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
"ninety47 Dukascopy toolbox".  If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem.hpp>
#include <ninety47/dukascopy.h>
#include <ninety47/dukascopy/defs.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
#include <pqxx/pqxx>
#include <set>
#include <string>

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

#ifndef TEST_DATA_PREFIX
#define TEST_DATA_PREFIX ..
#endif

namespace fs = boost::filesystem;
namespace pt = boost::posix_time;
namespace gr = boost::gregorian;


int main(void) {
	std::string currency("EURUSD");
	n47::tick_vector data;

	int unzipped = 0;
	int processed = 0;
	int already_loaded = 0;
	int empty_seen = 0;
	int couldnt_access = 0;
	int count = 0;

	pqxx::connection conn("postgresql://fx:fx@10.0.2.2/fx");

	pqxx::nontransaction txn(conn);

	std::string find_from("C:/VirtualBox/");
	std::string find_to("/media/sf_VirtualBox/");
	size_t str_pos = 0;

	std::set<std::string> filenames;
	pqxx::result res = txn.exec("SELECT filename FROM hours");

	for (const auto& row : res) {
		std::string replace_str = row["filename"].as<std::string>();

		str_pos = replace_str.find(find_from);
		if (str_pos != std::string::npos) {
			replace_str.replace(str_pos, find_from.length(), find_to);
		}

		filenames.insert(replace_str);
	}

	std::set<std::string> empty_filenames;
	pqxx::result res2 = txn.exec("SELECT filename FROM empty_hours");

	for (const auto& row2 : res2) {
		std::string replace_str = row2["filename"].as<std::string>();

		str_pos = replace_str.find(find_from);
		if (str_pos != std::string::npos) {
			replace_str.replace(str_pos, find_from.length(), find_to);
		}

		empty_filenames.insert(replace_str);
	}

	for(int year = 2003; year < 2025; ++year) {
		for(int month = 0; month < 12; ++month) {
			for(int day = 1; day < 32; ++day) {
				// November, April, June, and September are the months with 30 days , February has 28 days (29 days in the leap year)
				if(month == 10 || month == 3 || month == 5 || month == 8) {
					if(day > 30) {
						continue;
					}
				} else if(month == 1) {
					if(year % 4 == 0) {
						if(day > 29) {
							continue;
						}
					} else {
						if(day > 28) {
							continue;
						}
					}
				}

				for(int hour = 0; hour < 24; ++hour) {
					count++;

					if(count % 1000 == 0) {
						std::cout << "count: " << count << ", processed: " << processed << ", already_loaded: " << already_loaded
								<< ", empty_seen: " << empty_seen << ", couldnt_access: " << couldnt_access << std::endl;
					}

					size_t buffer_size;
					size_t buffer_size2;
					int counter;
					size_t raw_size = 0;

					char ofilename[256];
					memset(ofilename, 0, 256);
					snprintf(ofilename, 255, "/media/sf_VirtualBox/tickstory/%s_csv/%4d/%02d/%02d/%02dh_ticks.csv", currency.c_str(), year, month, day, hour);

					std::set<std::string>::iterator find_iter = filenames.find(ofilename);
					if(find_iter != filenames.end()) {
						// std::cout << "INFO  - output file already loaded: " << ofilename << std::endl;
						already_loaded++;
						continue;
					}

					char filename[256];
					memset(filename, 0, 256);
					snprintf(filename, 255, "/media/sf_VirtualBox/tickstory/%s/%4d/%02d/%02d/%02dh_ticks.bi5", currency.c_str(), year, month, day, hour);

					std::set<std::string>::iterator empty_find_iter = empty_filenames.find(filename);
					if(empty_find_iter != empty_filenames.end()) {
						// std::cout << "INFO  - empty data file already seen: " << filename << std::endl;
						empty_seen++;
						continue;
					}

					fs::path opath(ofilename);
					if (fs::exists(opath)) {
						std::cout << "WARN  - output file already exists: " << ofilename << std::endl;
						continue;
					}

					fs::path p(filename);
					if (!fs::exists(p) || !fs::is_regular(p)) {
						std::cout << "WARN  - couldn't access the data file: " << filename <<  std::endl;
						couldnt_access++;
						continue;
					}

					buffer_size = fs::file_size(p);

					if(buffer_size == 0) {
						std::cout << "ERROR - empty data file: " << filename << std::endl;
						continue;
					}

					unsigned char buffer[buffer_size];

					memset(ofilename, 0, 256);
					snprintf(ofilename, 255, "/media/sf_VirtualBox/tickstory/%s_csv", currency.c_str());
					boost::filesystem::path dir1(ofilename);
					if(!(boost::filesystem::exists(dir1))){
						if (boost::filesystem::create_directory(dir1))
							std::cout << "INFO  - created directory: " << ofilename << std::endl;
						else {
							std::cout << "ERROR - failed to created directory: " << ofilename << std::endl;
							continue;
						}
					}

					memset(ofilename, 0, 256);
					snprintf(ofilename, 255, "/media/sf_VirtualBox/tickstory/%s_csv/%4d", currency.c_str(), year);
					boost::filesystem::path dir2(ofilename);
					if(!(boost::filesystem::exists(dir2))){
						if (boost::filesystem::create_directory(dir2))
							std::cout << "INFO  - created directory: " << ofilename << std::endl;
						else {
							std::cout << "ERROR - failed to created directory: " << ofilename << std::endl;
							continue;
						}
					}

					memset(ofilename, 0, 256);
					snprintf(ofilename, 255, "/media/sf_VirtualBox/tickstory/%s_csv/%4d/%02d", currency.c_str(), year, month);
					boost::filesystem::path dir3(ofilename);
					if(!(boost::filesystem::exists(dir3))){
						if (boost::filesystem::create_directory(dir3))
							std::cout << "INFO  - created directory: " << ofilename << std::endl;
						else {
							std::cout << "ERROR - failed to created directory: " << ofilename << std::endl;
							continue;
						}
					}

					memset(ofilename, 0, 256);
					snprintf(ofilename, 255, "/media/sf_VirtualBox/tickstory/%s_csv/%4d/%02d/%02d", currency.c_str(), year, month, day);
					boost::filesystem::path dir4(ofilename);
					if(!(boost::filesystem::exists(dir4))){
						if (boost::filesystem::create_directory(dir4))
							std::cout << "INFO  - created directory: " << ofilename << std::endl;
						else {
							std::cout << "ERROR - failed to created directory: " << ofilename << std::endl;
							continue;
						}
					}

					std::ifstream fin;
					fin.open(filename, std::ifstream::binary);
					fin.read(reinterpret_cast<char*>(buffer), buffer_size);
					fin.close();

					data.clear();
					n47::read_bi5(data,
							buffer, buffer_size, PV_YEN_PAIR, &raw_size);
					n47::tick_vector::const_iterator iter;

					bool didUnzip = false;

					if (data.size() == 0) {
						std::cout << "ERROR - Failed to load the data: " << filename << std::endl;

						char filename2[256];
						memset(filename2, 0, 256);
						snprintf(filename2, 255, "/media/sf_VirtualBox/tickstory/%s/%4d/%02d/%02d/%02dh_ticks", currency.c_str(), year, month, day, hour);

						fs::path p2(filename2);
						if (!fs::exists(p2) || !fs::is_regular(p2)) {
							std::cout << "WARN  - couldn't access the decompressed file: " << filename2 <<  std::endl;

							char data_dir[256];
							memset(data_dir, 0, 256);
							snprintf(data_dir, 255, "/media/sf_VirtualBox/tickstory/%s/%4d/%02d/%02d", currency.c_str(), year, month, day);

							char bi5_filename[256];
							memset(bi5_filename, 0, 256);
							snprintf(bi5_filename, 255, "%02dh_ticks.bi5", hour);

							char extract_script[256];
							memset(extract_script, 0, 256);
							snprintf(extract_script, 255, "/usr/local/bin/extract_bi5");

							pid_t pid = fork();

							if (pid == 0) {
								char *args[] = {extract_script, data_dir, bi5_filename, NULL};
								execv(extract_script, args);
								std::cout << "ERROR  - execv failed" <<  std::endl;
								exit(1);
							} else if (pid > 0) {
								int status;
								waitpid(pid, &status, 0);
							} else {
								std::cout << "fork failed" << std::endl;
								continue;
							}

							fs::path p3(filename2);
							if (!fs::exists(p3) || !fs::is_regular(p3)) {
								std::cout << "WARN  - couldn't access the decompressed after unzip: " << filename2 <<  std::endl;
								continue;
							}
						}

						buffer_size2 = fs::file_size(p2);
						unsigned char buffer2[buffer_size2];
						raw_size = buffer_size2;

						std::ifstream fin2;
						fin2.open(filename2, std::ifstream::binary);
						fin2.read(reinterpret_cast<char*>(buffer2), buffer_size2);
						fin2.close();

						read_bin(data, buffer2, buffer_size2, PV_YEN_PAIR);

						if (data.size() == 0) {
							std::cout << "ERROR - Failed to load the data from decompressed file: " << filename << std::endl;
							continue;
						}

						didUnzip = true;
					}

					if (data.size() != (raw_size / n47::ROW_SIZE)) {
						std::cout << "ERROR - Loaded " << data.size()
								  << " ticks but file size indicates we should have loaded "
								  << (raw_size / n47::ROW_SIZE) << std::endl;
						continue;
					}

					memset(ofilename, 0, 256);
					snprintf(ofilename, 255, "/media/sf_VirtualBox/tickstory/%s_csv/%4d/%02d/%02d/%02dh_ticks.csv", currency.c_str(), year, month, day, hour);

					std::ofstream fout;
					fout.open(ofilename);
					if (!fout.is_open()) {
						std::cout << "ERROR - could not open output file: " << ofilename << std::endl;
						continue;
					}

					processed++;
					if(didUnzip) {
						unzipped++;
					}

					counter = 0;
					for (iter = data.begin(); iter != data.end(); iter++) {
						fout << iter->td << ", "
								  << iter->bid << ", " << iter->bidv << ", "
								  << iter->ask << ", " << iter->askv << std::endl;
						counter++;
					}

					fout.close();

					std::cout << "INFO  - wrote " << counter << " records to csv: " << ofilename << std::endl;
				}
			}
		}
	}

	std::cout << "count: " << count << ", processed: " << processed << ", already_loaded: " << already_loaded
									<< ", empty_seen: " << empty_seen << ", couldnt_access: " << couldnt_access << std::endl;

	std::cout << "INFO - processed: " << processed << ", unzipped: " << unzipped <<  std::endl;
}
