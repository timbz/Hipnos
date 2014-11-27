TEMPLATE    = subdirs

BUILD_CBLAS_PLUGIN = $$(HIPNOS_BUILD_CBLAS_PLUGIN)
BUILD_CUDA_PLUGIN = $$(HIPNOS_BUILD_CUDA_PLUGIN)
BUILD_GSL_PLUGIN = $$(HIPNOS_BUILD_GSL_PLUGIN)
BUILD_ACML_PLUGIN = $$(HIPNOS_BUILD_ACML_PLUGIN)
BUILD_APPML_PLUGIN = $$(HIPNOS_BUILD_APPML_PLUGIN)

!isEmpty( BUILD_CUDA_PLUGIN ) {
    message( "Building cuda math plugins" )
    SUBDIRS += HipnosCudaMathPlugin \
               HipnosCudaSinglePrecisionMathPlugin
}
!isEmpty( BUILD_GSL_PLUGIN ) {
    message( "Building GSL math plugins" )
    SUBDIRS += HipnosGSLMathPlugin
}
!isEmpty( BUILD_CBLAS_PLUGIN ) {
    message( "Building cuda math plugins" )
    SUBDIRS += HipnosCBlasMathPlugin
}
!isEmpty( BUILD_ACML_PLUGIN ) {
    message( "Building AMD ACML math plugins" )
    SUBDIRS += HipnosACMLMathPlugin
}
!isEmpty( BUILD_APPML_PLUGIN ) {
    message( "Building AMD APPML math plugins" )
    SUBDIRS += HipnosAPPMLMathPlugin
}



