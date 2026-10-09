set(TargetName "Mesh")
project(${TargetName})

set(SOURCES 
	"src/Mesh.cpp"
	"src/Mesh.h"
)

#set a cache variable for log support
set(LogSupport OFF CACHE BOOL "Enable log support")

set(Standalone OFF CACHE BOOL "Enable standalone mode")

add_library(${TargetName} ${SOURCES})

if(LogSupport)
	target_compile_definitions(${TargetName} PRIVATE LOG_SUPPORT)
	target_link_libraries(${TargetName} PRIVATE Log)
endif()

if(Standalone)
	target_compile_definitions(${TargetName} PRIVATE STANDALONE)
else()
	target_link_libraries(${TargetName} PRIVATE Kya Renderer glm)
endif()

target_include_directories(${TargetName} PUBLIC "src")