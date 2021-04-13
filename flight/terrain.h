#pragma once
#include <vector>
#include "Vertex.h"
class Terrain
{
public:
	Terrain(int widthIn, int heightIn) {
		width = widthIn; height = heightIn; numOfVertexPerCol = 0;
	}

	void loadVertices(std::vector<std::vector<uint16_t>> verticesIn);
	void loadIndices();

public:
	int width;
	int height;

	uint32_t numOfVertexPerCol;

	std::vector<Vertex> verticesData;
	std::vector<uint32_t> indicesData;
};

