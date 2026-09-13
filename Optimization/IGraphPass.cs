using TokenVector.Inference.Runtime;

namespace TokenVector.Inference.Optimization
{
    public interface IGraphPass
    {
        string Name { get; }
        bool Run(ExecutionGraph graph);
    }
}
