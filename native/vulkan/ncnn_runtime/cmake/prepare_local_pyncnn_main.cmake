if(NOT DEFINED LADA_LOCAL_PYNCNN_MAIN_INPUT OR NOT DEFINED LADA_LOCAL_PYNCNN_MAIN_OUTPUT)
    message(FATAL_ERROR "LADA local pyncnn main input/output paths are required.")
endif()

file(READ "${LADA_LOCAL_PYNCNN_MAIN_INPUT}" _lada_local_pyncnn_main)

set(_lada_local_bind_decl "namespace py = pybind11;\n\nvoid bind_lada_local_runtime(pybind11::module_& m);")
string(REPLACE "namespace py = pybind11;" "${_lada_local_bind_decl}" _lada_local_pyncnn_main "${_lada_local_pyncnn_main}")

set(_lada_local_bind_call "    bind_lada_local_runtime(m);\n\n#ifdef VERSION_INFO")
string(REPLACE "#ifdef VERSION_INFO" "${_lada_local_bind_call}" _lada_local_pyncnn_main "${_lada_local_pyncnn_main}")

file(WRITE "${LADA_LOCAL_PYNCNN_MAIN_OUTPUT}" "${_lada_local_pyncnn_main}")
