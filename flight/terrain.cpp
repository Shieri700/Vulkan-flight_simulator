#include "terrain.h"

void Terrain::loadVertices(std::vector<std::vector<uint16_t>> verticesIn) {
	numOfVertexPerCol = verticesIn.size();

	
	for (int h = 0; h < verticesIn.size(); h++) {
		for (int w = 0; w < verticesIn[h].size(); w++) {
			float y = -verticesIn[h][w];

			Vertex vertex;
			float x = h - ((float)height / 2.f);
			float z = w - ((float)width / 2.f);

			vertex.pos = { x, y, z };
			vertex.texCoord = { (float)w / numOfVertexPerCol , 1.f - (float)h / numOfVertexPerCol };
			vertex.color = { 0.f, 1.0f, 1.0f };

			// calculate normal
			

			verticesData.push_back(vertex);
		}
	}

	for (int h = 1; h < verticesIn.size() - 1; h++) {
		for (int w = 1; w < verticesIn[h].size() - 1; w++) {
			// calculate normal
			float hx1 = verticesIn[h - 1][w];
			float hx2 = verticesIn[h + 1][w];
			float hz1 = verticesIn[h][w - 1];
			float hz2 = verticesIn[h][w + 1];

			verticesData[h * numOfVertexPerCol + w].normal = glm::normalize(glm::vec3(hx1 - hx2, 2, hz1 - hz2));
		}
	}
	
	
	/*
	for (int h = 0; h < verticesIn.size() - 1; h++) {
		for (int w = 0; w < verticesIn[h].size() - 1; w++) {
			Vertex v0;
			float x = h*3.f - ((float)height * 3.f / 2.f);
			float y = -verticesIn[h][w] + 20;
			float z = w * 3.f - ((float)width * 3.f / 2.f);

			v0.pos = { x, y, z };
			v0.texCoord = { (float)w / numOfVertexPerCol , 1.f - (float)h / numOfVertexPerCol };
			v0.color = { 0.f, 1.0f, 1.0f };

			

			Vertex v1;
			x = h * 3.f - ((float)height * 3.f / 2.f);
			y = -verticesIn[h][w + 1] + 20;
			z = w * 3.f + 3.f - ((float)width * 3.f / 2.f);

			v1.pos = { x, y, z };
			v1.texCoord = { ((float)w + 1.f) / numOfVertexPerCol , 1.f - (float)h / numOfVertexPerCol };
			v1.color = { 0.f, 1.0f, 1.0f };

			

			Vertex v2;
			x = h * 3.f + 3.f - ((float)height * 3.f / 2.f);
			y = -verticesIn[h + 1][w] + 20;
			z = w * 3.f - ((float)width * 3.f / 2.f);

			v2.pos = { x, y, z };
			v2.texCoord = { ((float)w) / numOfVertexPerCol , 1.f - ((float)h + 1.f) / numOfVertexPerCol };
			v2.color = { 0.f, 1.0f, 1.0f };

			verticesData.emplace_back(v1);
			verticesData.emplace_back(v0);
			verticesData.emplace_back(v2);

			Vertex v3;
			x = h * 3.f + 3.f - ((float)height * 3.f / 2.f);
			y = -verticesIn[h + 1][w] + 20;
			z = w * 3.f - ((float)width * 3.f / 2.f);

			v3.pos = { x, y, z };
			v3.texCoord = { ((float)w) / numOfVertexPerCol , 1.f - ((float)h + 1.f) / numOfVertexPerCol };
			v3.color = { 0.f, 1.0f, 1.0f };

			verticesData.emplace_back(v3);

			Vertex v4;
			x = h * 3.f + 3.f - ((float)height * 3.f / 2.f);
			y = -verticesIn[h + 1][w + 1] + 20;
			z = w * 3.f + 3.f - ((float)width * 3.f / 2.f);

			v4.pos = { x, y, z };
			v4.texCoord = { ((float)w + 1.f) / numOfVertexPerCol , 1.f - ((float)h + 1.f) / numOfVertexPerCol };
			v4.color = { 0.f, 1.0f, 1.0f };

			verticesData.emplace_back(v4);

			Vertex v5;
			x = h * 3.f - ((float)height * 3.f / 2.f);
			y = -verticesIn[h][w + 1] + 20;
			z = w * 3.f + 3.f - ((float)width * 3.f / 2.f);

			v5.pos = { x, y, z };
			v5.texCoord = { ((float)w + 1.f) / numOfVertexPerCol , 1.f - (float)h / numOfVertexPerCol };
			v5.color = { 0.f, 1.0f, 1.0f };

			verticesData.emplace_back(v5);
		}
	}
	*/
	
}

void Terrain::loadIndices() {
	/*
	for (auto i = 0; i < verticesData.size(); i++) {
		indicesData.emplace_back(i);
	}
	*/
	for (uint32_t h = 0; h < numOfVertexPerCol - 1; h++) {
		for (uint32_t w = 0; w < numOfVertexPerCol - 1; w++) {

			indicesData.push_back(h * numOfVertexPerCol + w + 1);
			indicesData.push_back(h * numOfVertexPerCol + w);
			indicesData.push_back((h + 1) * numOfVertexPerCol + w);

			indicesData.push_back((h + 1) * numOfVertexPerCol + w);
			indicesData.push_back((h + 1) * numOfVertexPerCol + w + 1);
			indicesData.push_back(h * numOfVertexPerCol + w + 1);

		}
	}
	
	
}
