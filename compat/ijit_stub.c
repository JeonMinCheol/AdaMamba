unsigned int iJIT_GetNewMethodID(void) {
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}
