#include "application_context.h"

namespace dorado::application {

ContextContainer& contexts() {
    static ContextContainer context_container{};
    return context_container;
}

}  // namespace dorado::application
