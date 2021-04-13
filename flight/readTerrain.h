#pragma once
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#define NOMINMAX
#include <windows.h>

inline std::vector<std::vector<int>> loadBMP(std::string filename) {
	std::ifstream file(filename, std::ios::binary);

	BITMAPFILEHEADER fileHeader;
	file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));

	BITMAPINFOHEADER infoHeader;
	file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

	assert(infoHeader.biBitCount == 8);

	int height = infoHeader.biHeight;
	int width = infoHeader.biWidth;
	
	file.seekg(fileHeader.bfOffBits);
	const int padding = (4 - width % 4) % 4;

	std::vector<std::vector<int>> d;

	for (int h = 0; h < height; h++) {
		std::vector<int> newV;
		d.push_back(newV);
		for (int w = 0; w < width; w++) {
			d[h].push_back(file.get());
		}
		file.seekg(padding, std::ios::cur);
	}

	return d;
}

