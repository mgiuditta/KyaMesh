#include "Mesh.h"
#include "ed3D.h"
#include "ed3D/ed3DG3D.h"
#include "edList.h"
#include "Log.h"
#include "renderer.h"
#include "port.h"
#include "port/vu1_emu.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <unordered_set>

#ifdef LOG_SUPPORT
#define MESH_LOG(level, format, ...) MY_LOG_CATEGORY("MeshLibrary", level, format, ##__VA_ARGS__)
//#define MESH_LOG_TRACE(level, format, ...) MY_LOG_CATEGORY("MeshLibrary", level, format, ##__VA_ARGS__)
#define MESH_LOG_TRACE(level, format, ...)
#else
#define MESH_LOG(level, format, ...)
#define MESH_LOG_TRACE(level, format, ...)
#endif

namespace Renderer
{
	namespace Kya
	{
		constexpr uint32_t gGifTagCopyCode = 0x6c018000;

		static MeshLibrary gMeshLibrary;

		using StripCache = std::unordered_map<const ed_3d_strip*, Renderer::Kya::G3D::Strip*>;
		static StripCache gStripCache;

		static std::unordered_map<const ed_3d_strip*, Renderer::Kya::G3D::Object*> gObjectCache;

		static Gif_Tag ExtractGifTagFromVifList(ed_3d_strip* pStrip, int index = 0)
		{
			// Pull the prim reg out from the gif packet, not a big fan of this.
			char* const pVifList = reinterpret_cast<char*>(pStrip) + pStrip->vifListOffset;
			edpkt_data* pPkt = reinterpret_cast<edpkt_data*>(pVifList);

			while (index > 0) {
				if (pPkt->asU32[0] == gVifEndCode) {
					index--;
				}

				pPkt++;
			}

			if (pPkt[1].asU32[3] != gGifTagCopyCode) {
				pPkt = reinterpret_cast<edpkt_data*>(pVifList);
			}

			assert(pPkt[1].asU32[3] == gGifTagCopyCode);

			uint8_t* const pGifPkt = LOAD_POINTER_CAST(uint8_t*, pPkt[1].asU32[1]);
			Gif_Tag gifTag;
			gifTag.setTag(pGifPkt, true);
			return gifTag;
		}

		static void EmplaceHierarchy(std::vector<G3D::Hierarchy>& hierarchies, ed_g3d_hierarchy* pHierarchy, const int heirarchyIndex, G3D* pParent)
		{
			assert(pHierarchy);

			G3D::Hierarchy& hierarchy = hierarchies.emplace_back();
			hierarchy.pHierarchy = pHierarchy;
			hierarchy.pParent = pParent; // Probably should have the cluster as its parent.

			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessHierarchy Processing hierarchy: {}", pHierarchy->hash.ToString());

			for (int i = 0; i < pHierarchy->lodCount; i++) {
				MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessHierarchy Processing lod: {}", i);

				ed3DLod* pLod = pHierarchy->aLods + i;

				hierarchy.ProcessLod(pLod, heirarchyIndex, i);
			}
		}

		enum class DrawMode {
			v12,
			v32
		};

		static DrawMode GetDrawMode(ed_3d_strip* pStrip)
		{
			if ((pStrip->flags & 0x400) != 0) {
				return DrawMode::v12;
			}

			return DrawMode::v32;
		}

		static std::atomic<int> gSmoothLevel = 0;

		// KYA_MESH_SMOOTH=<n> overrides the setting: each triangle edge is split into n segments and the new
		// points move onto a PN-triangle patch, so low-poly silhouettes round off. Below 2 keeps the original mesh.
		static int GetSmoothLevel()
		{
			static const char* pOverride = getenv("KYA_MESH_SMOOTH");
			const int value = pOverride ? atoi(pOverride) : gSmoothLevel.load();
			return value < 2 ? 0 : std::min(value, 4);
		}

		using Vertex = Renderer::GSVertexUnprocessedNormal;

		static glm::vec3 Position(const Vertex& vtx) { return { vtx.XYZFlags.fXYZ[0], vtx.XYZFlags.fXYZ[1], vtx.XYZFlags.fXYZ[2] }; }
		static glm::vec3 Normal(const Vertex& vtx) { return { vtx.normal.fNormal[0], vtx.normal.fNormal[1], vtx.normal.fNormal[2] }; }
		static uint32_t Bone(const Vertex& vtx) { return vtx.XYZFlags.flags & 0x7ff; }

		struct PNEdge
		{
			glm::vec3 b0; // control point next to the first corner
			glm::vec3 b1; // control point next to the second corner
			glm::vec3 n;  // quadratic normal control point
		};

		static PNEdge MakePNEdge(const glm::vec3& p0, const glm::vec3& n0, const glm::vec3& p1, const glm::vec3& n1, bool bCurved)
		{
			const glm::vec3 d = p1 - p0;
			const glm::vec3 nSum = n0 + n1;
			if (glm::dot(nSum, nSum) < 1e-6f) {
				return { p0 + d / 3.0f, p1 - d / 3.0f, n0 };
			}

			if (!bCurved || glm::dot(d, d) < 1e-12f) {
				return { p0 + d / 3.0f, p1 - d / 3.0f, glm::normalize(nSum) };
			}

			const glm::vec3 b0 = (2.0f * p0 + p1 - glm::dot(d, n0) * n0) / 3.0f;
			const glm::vec3 b1 = (2.0f * p1 + p0 + glm::dot(d, n1) * n1) / 3.0f;
			const float v = 2.0f * glm::dot(d, n0 + n1) / glm::dot(d, d);
			const glm::vec3 n = glm::normalize(n0 + n1 - v * d);
			return { b0, b1, n };
		}

		// A triangle can only be curved when all three corners sit on the same bone: positions are in bone space.
		static bool CanCurve(const Vertex& a, const Vertex& b, const Vertex& c)
		{
			if (Bone(a) == 0 || Bone(a) != Bone(b) || Bone(a) != Bone(c)) {
				return false;
			}

			for (const Vertex* pVtx : { &a, &b, &c }) {
				const float length = glm::length(Normal(*pVtx));
				if (length < 0.5f || length > 1.5f) {
					return false;
				}
			}

			return true;
		}

		static void SmoothMesh(Renderer::NativeVertexBufferData& data, const int level)
		{
			const size_t indexCount = data.GetIndexTail();
			const size_t vertexCount = data.GetVertexTail();
			if (indexCount == 0 || indexCount % 3 != 0) {
				return;
			}

			const Vertex* pVertices = data.vertex.buff;
			const uint16_t* pIndices = data.index.buff;

			// Edges touching a triangle that cannot curve stay straight, so the neighbours do not crack.
			// ponytail: keyed by position + bone within one strip; edges shared across strips may still crack.
			auto edgeKey = [&](uint16_t i0, uint16_t i1) {
				glm::vec3 a = Position(pVertices[i0]);
				glm::vec3 b = Position(pVertices[i1]);
				if (std::tie(a.x, a.y, a.z) > std::tie(b.x, b.y, b.z)) {
					std::swap(a, b);
				}

				char key[sizeof(float) * 6 + sizeof(uint32_t)];
				memcpy(key, &a, sizeof(a));
				memcpy(key + sizeof(a), &b, sizeof(b));
				const uint32_t bone = Bone(pVertices[i0]);
				memcpy(key + sizeof(a) * 2, &bone, sizeof(bone));
				return std::string(key, sizeof(key));
			};

			std::unordered_set<std::string> straightEdges;
			size_t curvedTriangles = 0;
			for (size_t t = 0; t < indexCount; t += 3) {
				const uint16_t* pTri = pIndices + t;
				if (CanCurve(pVertices[pTri[0]], pVertices[pTri[1]], pVertices[pTri[2]])) {
					curvedTriangles++;
					continue;
				}

				for (int e = 0; e < 3; e++) {
					straightEdges.insert(edgeKey(pTri[e], pTri[(e + 1) % 3]));
				}
			}

			if (curvedTriangles == 0) {
				return;
			}

			// Every curved triangle becomes level^2 triangles with its own (level+1)(level+2)/2 vertices.
			const size_t pointsPerTriangle = (level + 1) * (level + 2) / 2;
			const size_t triangleCount = indexCount / 3;
			const size_t newVertexCount = vertexCount + curvedTriangles * pointsPerTriangle;
			const size_t newIndexCount = (triangleCount - curvedTriangles) * 3 + curvedTriangles * level * level * 3;
			if (newVertexCount > 0xffff) {
				return;
			}

			std::vector<Vertex> vertices(pVertices, pVertices + vertexCount);
			std::vector<uint16_t> indices;
			vertices.reserve(newVertexCount);
			indices.reserve(newIndexCount);

			for (size_t t = 0; t < indexCount; t += 3) {
				const uint16_t* pTri = pIndices + t;
				const Vertex corners[3] = { pVertices[pTri[0]], pVertices[pTri[1]], pVertices[pTri[2]] };

				if (!CanCurve(corners[0], corners[1], corners[2])) {
					indices.insert(indices.end(), pTri, pTri + 3);
					continue;
				}

				glm::vec3 p[3];
				glm::vec3 n[3];
				for (int c = 0; c < 3; c++) {
					p[c] = Position(corners[c]);
					n[c] = glm::normalize(Normal(corners[c]));
				}

				PNEdge edges[3];
				for (int e = 0; e < 3; e++) {
					const int c1 = (e + 1) % 3;
					const bool bCurved = straightEdges.find(edgeKey(pTri[e], pTri[c1])) == straightEdges.end();
					edges[e] = MakePNEdge(p[e], n[e], p[c1], n[c1], bCurved);
				}

				// Bezier triangle with corners p0 (w), p1 (u), p2 (v).
				const glm::vec3 b300 = p[0], b030 = p[1], b003 = p[2];
				const glm::vec3 b210 = edges[0].b0, b120 = edges[0].b1;
				const glm::vec3 b021 = edges[1].b0, b012 = edges[1].b1;
				const glm::vec3 b102 = edges[2].b0, b201 = edges[2].b1;
				const glm::vec3 e = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0f;
				const glm::vec3 b111 = e + (e - (b300 + b030 + b003) / 3.0f) / 2.0f;
				const glm::vec3 n110 = edges[0].n, n011 = edges[1].n, n101 = edges[2].n;

				const uint16_t base = static_cast<uint16_t>(vertices.size());
				for (int row = 0; row <= level; row++) {
					for (int col = 0; col <= level - row; col++) {
						const float v = float(row) / float(level);
						const float u = float(col) / float(level);
						const float w = 1.0f - u - v;

						Vertex vtx = corners[0];
						const glm::vec3 pos = b300 * (w * w * w) + b030 * (u * u * u) + b003 * (v * v * v)
							+ b210 * (3.0f * w * w * u) + b120 * (3.0f * w * u * u) + b201 * (3.0f * w * w * v)
							+ b021 * (3.0f * u * u * v) + b102 * (3.0f * w * v * v) + b012 * (3.0f * u * v * v)
							+ b111 * (6.0f * w * u * v);
						const glm::vec3 normal = glm::normalize(n[0] * (w * w) + n[1] * (u * u) + n[2] * (v * v)
							+ n110 * (w * u) + n011 * (u * v) + n101 * (w * v));

						for (int i = 0; i < 3; i++) {
							vtx.XYZFlags.fXYZ[i] = pos[i];
							vtx.normal.fNormal[i] = normal[i];
						}

						vtx.normal.fNormal[3] = corners[0].normal.fNormal[3] * w + corners[1].normal.fNormal[3] * u + corners[2].normal.fNormal[3] * v;
						for (int i = 0; i < 2; i++) {
							vtx.STQ.ST[i] = static_cast<int32_t>(std::lround(float(corners[0].STQ.ST[i]) * w + float(corners[1].STQ.ST[i]) * u + float(corners[2].STQ.ST[i]) * v));
						}

						for (int i = 0; i < 4; i++) {
							vtx.RGBA[i] = static_cast<uint32_t>(std::lround(float(corners[0].RGBA[i]) * w + float(corners[1].RGBA[i]) * u + float(corners[2].RGBA[i]) * v));
						}

						vertices.push_back(vtx);
					}
				}

				// Index of the grid point at (row, col).
				auto at = [&](int row, int col) {
					return static_cast<uint16_t>(base + row * (level + 1) - row * (row - 1) / 2 + col);
				};

				for (int row = 0; row < level; row++) {
					for (int col = 0; col < level - row; col++) {
						indices.insert(indices.end(), { at(row, col), at(row, col + 1), at(row + 1, col) });
						if (col < level - row - 1) {
							indices.insert(indices.end(), { at(row, col + 1), at(row + 1, col + 1), at(row + 1, col) });
						}
					}
				}
			}

			_aligned_free(data.vertex.buff);
			_aligned_free(data.index.buff);
			data.Init(static_cast<int>(vertices.size()), static_cast<int>(indices.size()));
			memcpy(data.vertex.buff, vertices.data(), vertices.size() * sizeof(Vertex));
			memcpy(data.index.buff, indices.data(), indices.size() * sizeof(uint16_t));
			data.vertex.head = 0;
			data.vertex.tail = data.vertex.next = vertices.size();
			data.index.tail = indices.size();
		}
	}
}

constexpr const char* gDebugMeshName = "SECT1.g3d_17_0_0";

void Renderer::Kya::G3D::Strip::PreProcessVertices(int textureLayerIndex, SimpleMesh* pMesh) const
{
	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Object::Strip::PreProcessVertices Processing strip name: {}", pMesh->GetName());

	if (pMesh->GetName() == gDebugMeshName) {
		MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Object::Strip::PreProcessVertices Processing strip name: {}", pMesh->GetName());
	}

	const Gif_Tag firstGifTag = ExtractGifTagFromVifList(pStrip);

	const DrawMode drawMode = GetDrawMode(pStrip);

	auto& vertexBufferData = pMesh->GetVertexBufferData();

	// Assume that the first gif tag has the largest vtx count.
	int totalVtxCount = 0;

	for (int j = 0; j < pStrip->meshCount; j++) {
		Gif_Tag gifTag = ExtractGifTagFromVifList(pStrip, j);
		totalVtxCount += gifTag.nLoop;
	}

	assert(totalVtxCount > 0);

	vertexBufferData.Init(totalVtxCount * 2, totalVtxCount * 4);

	union VertexColor {
		uint32_t rgba;

		struct {
			uint8_t r;
			uint8_t g;
			uint8_t b;
			uint8_t a;
		};
	};

	static_assert(sizeof(VertexColor) == 4);

	union TextureData {
		uint32_t st;

		struct {
			int16_t s;
			int16_t t;
		};
	};

	static_assert(sizeof(TextureData) == 4);

	VertexColor* pRgba = LOAD_POINTER_CAST(VertexColor*, pStrip->pColorBuf);
	TextureData* pStq = LOAD_POINTER_CAST(TextureData*, pStrip->pSTBuf);
	pStq += 4;

	TextureData* pLayerStq = pStq;
	if (textureLayerIndex > 0) {
		int* pSTHeader = LOAD_POINTER_CAST(int*, pStrip->pSTBuf);
		const int stLayerStride = pSTHeader[1] * 4;
		pLayerStq = reinterpret_cast<TextureData*>(pSTHeader + textureLayerIndex * stLayerStride + 4);
	}

	// This increases by 2 every loop because we start the next vtx at the end of the previous vtx.
	int vtxOffset = 0;

	int meshOffset = 0;

	for (int j = 0; j < pStrip->meshCount; j++) {
		MESH_LOG_TRACE(LogLevel::Info, "Renderer::Kya::G3D::Strip::PreProcessVertices Starting section: {}", j);
		Gif_Tag gifTag = ExtractGifTagFromVifList(pStrip, j);

		for (int i = 0; i < gifTag.nLoop; i++) {
			const int index = i + meshOffset;
			const int adjustedIndex = index - vtxOffset;

			Renderer::GSVertexUnprocessedNormal vtx;
			vtx.RGBA[0] = pRgba[index].r;
			vtx.RGBA[1] = pRgba[index].g;
			vtx.RGBA[2] = pRgba[index].b;
			vtx.RGBA[3] = pRgba[index].a;

			const int stIndex = textureLayerIndex == 0 ? index : i + j * 0x14;
			vtx.STQ.ST[0] = pLayerStq[stIndex].s;
			vtx.STQ.ST[1] = pLayerStq[stIndex].t;
			vtx.STQ.Q = 1.0f;

			if (pStrip->pNormalBuf) {
				edVertexNormal* pNormal = LOAD_POINTER_CAST(edVertexNormal*, pStrip->pNormalBuf);

				vtx.normal.fNormal[0] = int15_to_float(pNormal[adjustedIndex].x);
				vtx.normal.fNormal[1] = int15_to_float(pNormal[adjustedIndex].y);
				vtx.normal.fNormal[2] = int15_to_float(pNormal[adjustedIndex].z);
				vtx.normal.fNormal[3] = int15_to_float(pNormal[adjustedIndex].pad);

				MESH_LOG_TRACE(LogLevel::Info, "Renderer::Kya::G3D::Strip::PreProcessVertices Processing vertex: {}, normal: ({}, {}, {})", i, vtx.normal.fNormal[0], vtx.normal.fNormal[1], vtx.normal.fNormal[2]);
			}
			else {
				memset(&vtx.normal, 0, sizeof(vtx.normal));
			}

			// Covert xyz 16
			if (drawMode == DrawMode::v12) {
				struct Vertex12 {
					int16_t x;
					int16_t y;
					int16_t z;
					int16_t flags;
				};

				Vertex12* pVertex = LOAD_POINTER_CAST(Vertex12*, pStrip->pVertexBuf);

				vtx.XYZFlags.fXYZ[0] = int12_to_float(pVertex[adjustedIndex].x);
				vtx.XYZFlags.fXYZ[1] = int12_to_float(pVertex[adjustedIndex].y);
				vtx.XYZFlags.fXYZ[2] = int12_to_float(pVertex[adjustedIndex].z);
				vtx.XYZFlags.flags = pVertex[adjustedIndex].flags;
			}
			else {
				GSVertexUnprocessed::Vertex* pVertex = LOAD_POINTER_CAST(GSVertexUnprocessed::Vertex*, pStrip->pVertexBuf);
				vtx.XYZFlags = pVertex[adjustedIndex];
			}

			const uint64_t primReg = firstGifTag.tag.PRIM;
			const GIFReg::GSPrim primPacked = *reinterpret_cast<const GIFReg::GSPrim*>(&primReg);

			const uint skip = vtx.XYZFlags.flags & 0x8000;

			MESH_LOG_TRACE(LogLevel::Info, "Renderer::Kya::G3D::Strip::PreProcessVertices Processing vertex: {}, drawMode: {}, primPacked: 0x{:x}, nloop: 0x{:x}, skip: 0x{:x}", i, (int)drawMode, primReg, gifTag.nLoop, skip);

			MESH_LOG_TRACE(LogLevel::Info, "Renderer::Kya::G3D::Strip::PreProcessVertices Processing vertex: {}, (S: {} T: {} Q: {}) (R: {} G: {} B: {} A: {}) (X: {} Y: {} Z: {} Skip: {})\n",
				i, vtx.STQ.ST[0], vtx.STQ.ST[1], vtx.STQ.Q, vtx.RGBA[0], vtx.RGBA[1], vtx.RGBA[2], vtx.RGBA[3], vtx.XYZFlags.fXYZ[0], vtx.XYZFlags.fXYZ[1], vtx.XYZFlags.fXYZ[2], vtx.XYZFlags.flags);

			Renderer::KickVertex(vtx, primPacked, skip, vertexBufferData);

			MESH_LOG_TRACE(LogLevel::Info, "Renderer::Kya::G3D::Strip::PreProcessVertices Kick complete vtx tail: 0x{:x} index tail: 0x{:x}",
				vertexBufferData.GetVertexTail(), vertexBufferData.GetIndexTail());
		}

		meshOffset += gifTag.nLoop;
		vtxOffset += 2;
	}

	if (GetSmoothLevel() > 0) {
		SmoothMesh(vertexBufferData, GetSmoothLevel());
	}

	//assert(internalVertexBuffer.GetIndexTail() > 0);
}

void Renderer::Kya::SetMeshSmoothLevel(int level)
{
	gSmoothLevel = level;
}

Renderer::SimpleMesh* Renderer::Kya::G3D::Strip::GetSimpleMesh(int textureLayerIndex) const
{
	if (textureLayerIndex <= 0) {
		return pSimpleMesh.get();
	}

	const size_t layerIndex = static_cast<size_t>(textureLayerIndex);
	if (layerSimpleMeshes.size() <= layerIndex) {
		layerSimpleMeshes.resize(layerIndex + 1);
	}

	if (!layerSimpleMeshes[layerIndex]) {
		std::string meshName = pSimpleMesh->GetName();
		meshName += "_layer_";
		meshName += std::to_string(textureLayerIndex);

		layerSimpleMeshes[layerIndex] = std::make_unique<SimpleMesh>(meshName, pSimpleMesh->GetPrim(), pSimpleMesh->GetStripFlags());
		PreProcessVertices(textureLayerIndex, layerSimpleMeshes[layerIndex].get());
	}

	return layerSimpleMeshes[layerIndex].get();
}

void Renderer::Kya::G3D::Cluster::ProcessStrip(ed_3d_strip* pStrip, const int stripIndex)
{
	assert(pStrip);

	assert(pStrip->meshCount > 0);

	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Cluster::ProcessStrip Processing strip flags: 0x{:x}", pStrip->flags);

	Strip& strip = strips.emplace_back();
	strip.pStrip = pStrip;
	strip.pParent = this;

	const Gif_Tag gifTag = ExtractGifTagFromVifList(pStrip);

	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Cluster::ProcessStrip Processing strip gifTag: NLOOP 0x{:x} NREG 0x{:x} PRIM 0x{:x}", (uint)gifTag.tag.NLOOP, (uint)gifTag.tag.NREG, (uint)gifTag.tag.PRIM);
	const uint64_t primReg = gifTag.tag.PRIM;
	const GIFReg::GSPrim prim = *reinterpret_cast<const GIFReg::GSPrim*>(&primReg);

	// strip everything before the last forward slash
	std::string meshName = this->pParent->GetName().substr(this->pParent->GetName().find_last_of('\\') + 1);
	meshName += "_";
	meshName += std::to_string(stripIndex);

	strip.pSimpleMesh = std::make_unique<SimpleMesh>(meshName, prim, pStrip->flags);

	strip.PreProcessVertices(0, strip.pSimpleMesh.get());
}

void Renderer::Kya::G3D::Cluster::CacheStrips()
{
	for (auto& strip : strips) {
		gStripCache[strip.pStrip] = &strip;
	}
}

void Renderer::Kya::G3D::Cluster::ProcessHierarchy(ed_g3d_hierarchy* pHierarchy, const int heirarchyIndex)
{
	EmplaceHierarchy(hierarchies, pHierarchy, heirarchyIndex, pParent);
}

void Renderer::Kya::G3D::Object::ProcessStrip(ed_3d_strip* pStrip, const int heirarchyIndex, const int lodIndex, const int stripIndex)
{
	assert(pStrip);

	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Object::ProcessStrip Processing strip flags: 0x{:x}", pStrip->flags);

	Strip& strip = strips.emplace_back();
	strip.pStrip = pStrip;
	strip.pParent = this;

	Gif_Tag gifTag = ExtractGifTagFromVifList(pStrip);

	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Object::ProcessStrip Processing strip gifTag: NLOOP 0x{:x} NREG 0x{:x} PRIM 0x{:x}", (uint)gifTag.tag.NLOOP, (uint)gifTag.tag.NREG, (uint)gifTag.tag.PRIM);
	const uint64_t primReg = gifTag.tag.PRIM;
	const GIFReg::GSPrim prim = *reinterpret_cast<const GIFReg::GSPrim*>(&primReg);

	// strip everything before the last forward slash
	std::string meshName = this->pParent ? this->pParent->pParent->pParent->GetName().substr(this->pParent->pParent->pParent->GetName().find_last_of('\\') + 1) :
		"None";
	meshName += "_";
	meshName += std::to_string(heirarchyIndex);
	meshName += "_";
	meshName += std::to_string(lodIndex);
	meshName += "_";
	meshName += std::to_string(stripIndex);

	strip.pSimpleMesh = std::make_unique<SimpleMesh>(meshName, prim, pStrip->flags);

	strip.PreProcessVertices(0, strip.pSimpleMesh.get());
}

void Renderer::Kya::G3D::Hierarchy::Lod::Object::CacheStrips()
{
	for (auto& strip : strips) {
		gStripCache[strip.pStrip] = &strip;
	}
}

void Renderer::Kya::G3D::Lod::ProcessObject(ed_g3d_object* pObject, const int heirarchyIndex, const int lodIndex)
{
	object.pObject = pObject;
	object.pParent = this;

	if (pObject->p3DData) {
		ed_3d_strip* pStrip = LOAD_POINTER_CAST(ed_3d_strip*, pObject->p3DData);
		int stripIndex = 0;

		while (stripIndex < pObject->stripCount) {
			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Hierarchy::Lod::ProcessObject Processing strip: {}", stripIndex);

			object.ProcessStrip(pStrip, heirarchyIndex, lodIndex, stripIndex);
			pStrip = LOAD_POINTER_CAST(ed_3d_strip*, pStrip->pNext);
			stripIndex++;
		}

		object.CacheStrips();
	}
}

void Renderer::Kya::G3D::Hierarchy::ProcessLod(ed3DLod* pLod, const int heirarchyIndex, const int lodIndex)
{
	assert(pLod);

	if (pLod->pObj) {
		Lod& lod = lods.emplace_back();
		lod.pLod = pLod;
		lod.pParent = this;

		ed_hash_code* pHash = LOAD_POINTER_CAST(ed_hash_code*, pLod->pObj);
		MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Hierarchy::ProcessLod Processing lod: {}", pHash->hash.ToString());

		ed_Chunck* pOBJ = LOAD_POINTER_CAST(ed_Chunck*, pHash->pData);

		if (pOBJ) {
			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Hierarchy::ProcessLod Object chunk header: {}", pOBJ->GetHeaderString());

			ed_g3d_object* pObject = reinterpret_cast<ed_g3d_object*>(pOBJ + 1);
			lod.ProcessObject(pObject, heirarchyIndex, lodIndex);
		}
	}
	else {
		MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Hierarchy::ProcessLod No lod data");
	}
}

Renderer::Kya::G3D::G3D(ed_g3d_manager* pManager, std::string name)
	: pManager(pManager)
	, name(name)
{
	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::G3D Beginning processing of mesh: {}", name.c_str());

	if (pManager->HALL) {
		ProcessHALL();
	}

	if (pManager->CSTA) {
		ProcessCSTA();
	}
}

void Renderer::Kya::G3D::ProcessHierarchy(ed_g3d_hierarchy* pHierarchy, const int heirarchyIndex)
{
	EmplaceHierarchy(hierarchies, pHierarchy, heirarchyIndex, this);
}

void Renderer::Kya::G3D::ProcessHALL()
{
	assert(pManager->HALL);

	ed_Chunck* pHASH = pManager->HALL + 1;
	ed_hash_code* pHashCode = reinterpret_cast<ed_hash_code*>(pHASH + 1);
	const int chunkNb = edChunckGetNb(pHASH, reinterpret_cast<char*>(pManager->HALL) + pManager->HALL->size);

	hierarchies.reserve(chunkNb);
	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::G3D Nb Chunks: {}", chunkNb);

	for (int curIndex = 0; curIndex < chunkNb - 1; curIndex = curIndex + 1) {
		MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::G3D Processing hierarchy {}/{} hash: {}", curIndex, chunkNb, pHashCode->hash.ToString());

		ed_g3d_hierarchy* pHierarchy = ed3DG3DHierarchyGetFromIndex(pManager, curIndex);

		if (pHierarchy) {
			ProcessHierarchy(pHierarchy, curIndex);
		}

		pHashCode++;
	}
}

void Renderer::Kya::G3D::ProcessCluster(ed_g3d_cluster* pCluster)
{
	assert(pCluster);

	cluster.pData = pCluster;
	cluster.pParent = this;

	const uint stripCountArrayEntryIndex = 4;

	const uint stripCount = pCluster->aClusterStripCounts[stripCountArrayEntryIndex];

	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCluster Processing CDQU chunk stripCount: {}", stripCount);

	bool bProcessedStrip = false;

	if ((stripCount != 0) && (bProcessedStrip = true, stripCount != 0)) {
		ed_Chunck* pMBNK = LOAD_POINTER_CAST(ed_Chunck*, pCluster->clusterDetails.pMBNK);
		ed_3d_strip* p3DStrip = LOAD_POINTER_CAST(ed_3d_strip*, pCluster->p3DStrip);

		uint stripIndex = 0;

		while (stripIndex < stripCount) {
			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::Hierarchy::Lod::ProcessObject Processing strip: {}", stripIndex);

			cluster.ProcessStrip(p3DStrip, stripIndex);
			p3DStrip = LOAD_POINTER_CAST(ed_3d_strip*, p3DStrip->pNext);
			stripIndex++;
		}

		cluster.CacheStrips();

		bProcessedStrip = true;
	}

	uint spriteCount = pCluster->spriteCount;

	if (spriteCount != 0) {

	}

	uint clusterHierCount = pCluster->clusterHierCount;

	if (clusterHierCount != 0) {
		ed_Chunck* pHASH = reinterpret_cast<ed_Chunck*>(pCluster + 1);
		ed_hash_code* pHashCode = reinterpret_cast<ed_hash_code*>(pHASH + 1);

		for (int i = 0; i < clusterHierCount; i++) {
			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCluster Processing cluster hierarchy: {}", pHashCode->hash.ToString());

			ed_Chunck* pHIER = LOAD_POINTER_CAST(ed_Chunck*, pHashCode->pData);
			assert(pHIER->hash == HASH_CODE_HIER);

			ed_g3d_hierarchy* pHierarchy = reinterpret_cast<ed_g3d_hierarchy*>(pHIER + 1);

			if (pHierarchy) {
				cluster.ProcessHierarchy(pHierarchy, i);
			}

			pHashCode++;
		}
	}
}

void Renderer::Kya::G3D::ProcessCSTA()
{
	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCSTA CSTA chunk header: {}", pManager->CSTA->GetHeaderString());

	ed_Chunck* pClusterTypeChunk = pManager->CSTA + 1;

	MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCSTA Cluster chunk header: {}", pClusterTypeChunk->GetHeaderString());

	if (pClusterTypeChunk->hash == HASH_CODE_CDOA) {
		assert(false);
		MeshData_CSTA* pCSTA = reinterpret_cast<MeshData_CSTA*>(pClusterTypeChunk + 1);


	}
	else {
		if (pClusterTypeChunk->hash == HASH_CODE_CDQA) {
			MeshData_CSTA* pCSTA = reinterpret_cast<MeshData_CSTA*>(pClusterTypeChunk + 1);

			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCSTA Processing CDQA chunk");

			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCSTA field_0x20: {}", pCSTA->field_0x20.ToString());
			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCSTA worldLocation: {}", pCSTA->worldLocation.ToString());

			ed_Chunck* pCDQU = edChunckGetFirst(pCSTA + 1, reinterpret_cast<char*>(pClusterTypeChunk) + pClusterTypeChunk->size);
			MESH_LOG(LogLevel::Info, "Renderer::Kya::G3D::ProcessCSTA CDQA chunk header: {}", pCDQU->GetHeaderString());

			if (pCDQU) {
				ed_g3d_cluster* pCDQUData = reinterpret_cast<ed_g3d_cluster*>(pCDQU + 1);
				ProcessCluster(pCDQUData);
			}
		}
	}
}

void Renderer::Kya::MeshLibrary::Init()
{
	ed3DGetMeshLoadedDelegate() += Renderer::Kya::MeshLibrary::AddMesh;
}

const Renderer::Kya::G3D::Strip* Renderer::Kya::MeshLibrary::FindStrip(const ed_3d_strip* pStrip) const
{
	constexpr bool bUseStripCache = true;

	if (bUseStripCache) {
		assert(gStripCache.find(pStrip) != gStripCache.end());
		return gStripCache[pStrip];
	}

	int hierarchyIndex = 0;
	int lodIndex = 0;
	int stripIndex = 0;

	for (const auto& mesh : gMeshes) {
		for (const auto& hierarchy : mesh.GetHierarchies()) {
			for (const auto& lod : hierarchy.lods) {
				for (const auto& strip : lod.object.strips) {
					if (strip.pStrip == pStrip) {
						MESH_LOG(LogLevel::Info, "Renderer::Kya::MeshLibrary::FindStrip Found hierarchy: {}, lod: {}, strip: {}", hierarchyIndex, lodIndex, stripIndex);
						return &strip;
					}

					stripIndex++;
				}
			}

			lodIndex++;
		}

		hierarchyIndex++;
	}

	MESH_LOG(LogLevel::Info, "Renderer::Kya::MeshLibrary::FindStrip Strip not found");
	return nullptr;
}

void Renderer::Kya::MeshLibrary::RenderNode(const edNODE* pNode, int textureLayerIndex) const
{
	ed_3d_strip* pStrip = reinterpret_cast<ed_3d_strip*>(pNode->pData);

	const G3D::Strip* pRendererStrip = FindStrip(pStrip);
	assert(pRendererStrip);

	if (pRendererStrip) {
		Renderer::SimpleMesh* pSimpleMesh = pRendererStrip->GetSimpleMesh(textureLayerIndex);
		if (pSimpleMesh) {
			Renderer::RenderMesh(pSimpleMesh, pNode->header.typeField.flags);
		}
		else {
			MESH_LOG(LogLevel::Error, "Renderer::Kya::MeshLibrary::RenderNode no simple mesh available for rendering layer {}", textureLayerIndex);
		}
	}
	else {
		MESH_LOG(LogLevel::Error, "Renderer::Kya::MeshLibrary::RenderNode Strip not found");
	}
}

void Renderer::Kya::MeshLibrary::CacheDlistStrip(ed_3d_strip* pStrip)
{
	if (gObjectCache.find(pStrip) == gObjectCache.end()) {
		gObjectCache[pStrip] = new Renderer::Kya::G3D::Object();
	}

	auto* pObj = gObjectCache[pStrip];
	pObj->strips.clear();
	pObj->ProcessStrip(pStrip, 0, 0, 0);
	pObj->CacheStrips();
}

void Renderer::Kya::MeshLibrary::AddMesh(ed_g3d_manager* pManager, std::string name)
{
	gMeshLibrary.gMeshes.emplace_back(pManager, name);
}

const Renderer::Kya::MeshLibrary& Renderer::Kya::GetMeshLibrary()
{
	return gMeshLibrary;
}

Renderer::Kya::MeshLibrary& Renderer::Kya::GetMeshLibraryMutable()
{
	return gMeshLibrary;
}
