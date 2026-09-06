#ifndef _FILO2_PARSER_HPP_
#define _FILO2_PARSER_HPP_

#include <optional>
#include <string>

#include "base/NonCopyable.hpp"
#include "InstanceData.hpp"

namespace cobra {

    // Very simple TSPLIB-like parser specialized to parse X-like instances.
    class Parser : private NonCopyable<Parser> {
    public:
        using Data = InstanceData;

        Parser(const std::string& filepath);

        // Parses the instance and returns the parsed data if successful, nullopt otherwise.
        std::optional<InstanceData> Parse();

    private:
        // Instance file path.
        const std::string filepath;
    };

}  // namespace cobra

#endif
