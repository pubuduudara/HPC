

void checkError(cl_int error, char *str)
{
    if (error == 0)
        return;
    fprintf(stderr, "%s: ", str);
    // Print error message
    switch (error)
    {
    case -1:
        fprintf(stderr, "CL_DEVICE_NOT_FOUND ");
        break;
    case -2:
        fprintf(stderr, "CL_DEVICE_NOT_AVAILABLE ");
        break;
    case -3:
        fprintf(stderr, "CL_COMPILER_NOT_AVAILABLE ");
        break;
    case -4:
        fprintf(stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE ");
        break;
    case -5:
        fprintf(stderr, "CL_OUT_OF_RESOURCES ");
        break;
    case -6:
        fprintf(stderr, "CL_OUT_OF_HOST_MEMORY ");
        break;
    case -7:
        fprintf(stderr, "CL_PROFILING_INFO_NOT_AVAILABLE ");
        break;
    case -8:
        fprintf(stderr, "CL_MEM_COPY_OVERLAP ");
        break;
    case -9:
        fprintf(stderr, "CL_IMAGE_FORMAT_MISMATCH ");
        break;
    case -10:
        fprintf(stderr, "CL_IMAGE_FORMAT_NOT_SUPPORTED ");
        break;
    case -11:
        fprintf(stderr, "CL_BUILD_PROGRAM_FAILURE ");
        break;
    case -12:
        fprintf(stderr, "CL_MAP_FAILURE ");
        break;
    case -13:
        fprintf(stderr, "CL_MISALIGNED_SUB_BUFFER_OFFSET ");
        break;
    case -14:
        fprintf(stderr, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
        break;

    case -30:
        fprintf(stderr, "CL_INVALID_VALUE ");
        break;
    case -31:
        fprintf(stderr, "CL_INVALID_DEVICE_TYPE ");
        break;
    case -32:
        fprintf(stderr, "CL_INVALID_PLATFORM ");
        break;
    case -33:
        fprintf(stderr, "CL_INVALID_DEVICE ");
        break;
    case -34:
        fprintf(stderr, "CL_INVALID_CONTEXT ");
        break;
    case -35:
        fprintf(stderr, "CL_INVALID_QUEUE_PROPERTIES ");
        break;
    case -36:
        fprintf(stderr, "CL_INVALID_COMMAND_QUEUE ");
        break;
    case -37:
        fprintf(stderr, "CL_INVALID_HOST_PTR ");
        break;
    case -38:
        fprintf(stderr, "CL_INVALID_MEM_OBJECT ");
        break;
    case -39:
        fprintf(stderr, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
        break;
    case -40:
        fprintf(stderr, "CL_INVALID_IMAGE_SIZE ");
        break;
    case -41:
        fprintf(stderr, "CL_INVALID_SAMPLER ");
        break;
    case -42:
        fprintf(stderr, "CL_INVALID_BINARY ");
        break;
    case -43:
        fprintf(stderr, "CL_INVALID_BUILD_OPTIONS ");
        break;
    case -44:
        fprintf(stderr, "CL_INVALID_PROGRAM ");
        break;
    case -45:
        fprintf(stderr, "CL_INVALID_PROGRAM_EXECUTABLE ");
        break;
    case -46:
        fprintf(stderr, "CL_INVALID_KERNEL_NAME ");
        break;
    case -47:
        fprintf(stderr, "CL_INVALID_KERNEL_DEFINITION ");
        break;
    case -48:
        fprintf(stderr, "CL_INVALID_KERNEL ");
        break;
    case -49:
        fprintf(stderr, "CL_INVALID_ARG_INDEX ");
        break;
    case -50:
        fprintf(stderr, "CL_INVALID_ARG_VALUE ");
        break;
    case -51:
        fprintf(stderr, "CL_INVALID_ARG_SIZE ");
        break;
    case -52:
        fprintf(stderr, "CL_INVALID_KERNEL_ARGS ");
        break;
    case -53:
        fprintf(stderr, "CL_INVALID_WORK_DIMENSION ");
        break;
    case -54:
        fprintf(stderr, "CL_INVALID_WORK_GROUP_SIZE ");
        break;
    case -55:
        fprintf(stderr, "CL_INVALID_WORK_ITEM_SIZE ");
        break;
    case -56:
        fprintf(stderr, "CL_INVALID_GLOBAL_OFFSET ");
        break;
    case -57:
        fprintf(stderr, "CL_INVALID_EVENT_WAIT_LIST ");
        break;
    case -58:
        fprintf(stderr, "CL_INVALID_EVENT ");
        break;
    case -59:
        fprintf(stderr, "CL_INVALID_OPERATION ");
        break;
    case -60:
        fprintf(stderr, "CL_INVALID_GL_OBJECT ");
        break;
    case -61:
        fprintf(stderr, "CL_INVALID_BUFFER_SIZE ");
        break;
    case -62:
        fprintf(stderr, "CL_INVALID_MIP_LEVEL ");
        break;
    case -63:
        fprintf(stderr, "CL_INVALID_GLOBAL_WORK_SIZE ");
        break;
    default:
        fprintf(stderr, "UNRECOGNIZED ERROR CODE (%d)", error);
    }
    fprintf(stderr, "\n");
}