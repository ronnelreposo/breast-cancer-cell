(* drop out *)

open System.IO;
open System.Text.RegularExpressions;

type Vector = Vector of float list
type Matrix = Matrix of float list list


/// Matrix Transpose.
let transpose xss =
 let rec f xss acc =
  match xss with
  | [] -> failwith "xss must not contain empty vectors."
  | hd::_ ->
   match hd with
   | [] -> List.rev acc
   | _ ->
    f <| List.map (List.skip 1) xss <| (List.map List.head xss)::acc
 f xss List.empty

///Operation of two Vectors.
let vecOp f xs ys =
 match xs, ys with
 | [], [] -> []
 | [], hd::tl -> ys
 | a::b, [] -> xs
 | a::b, c::d -> List.map2 f xs ys

/// The Dot Product of xs and ys.
let dot xs ys = List.map2 (*) xs ys |> List.sum

let square x = x * x

let distance xs ys =
  (List.map2 (-) xs ys)
  |> List.map square
  |> List.average
  |> sqrt
  |> (*) 0.5

///Shuffle List (Fisher Yates Alogrithm).
let shuffle xs =
 let f (rand: System.Random) (xs:List<'a>) =
  let rec shuffleTo (indexes: int[]) upTo =
   match upTo with
   | 0 -> indexes
   | _ ->
    let fst = rand.Next(upTo)
    let temp = indexes.[fst]
    indexes.[fst] <- indexes.[upTo]
    indexes.[upTo] <- temp
    shuffleTo indexes (upTo - 1)
  let length = xs.Length
  let indexes = [| 0 .. length - 1 |]
  let shuffled = shuffleTo indexes (length - 1)
  List.permute (fun i -> shuffled.[i]) xs
 f (System.Random()) xs

/// Maps the elements the first list (xs) to second list (ys) using the mapper function.
/// first, it gets the first element of first list (xs) and maps to second list (ys)
/// using the mapper function. i.e. (List.map (f x) ys).
/// Finally it returns the accumulated mapped list.
/// mapToSecondList (+) ["1"; "2"; "3"] ["2"; "3", "4"] =
/// [ ["12"; "13"; "14"]; ["22"; "23"; "24"]; ["32"; "33"; "34"] ].
let mapToSecondList f xs ys =
    let mapEach x = List.map (f x) ys
    List.map mapEach xs

/// Scalar Vector Multiplication.
let smul c = List.map ((*) c)

/// Vector Multiplication.
let mul = List.map2 (*)

/// Vector Addition.
let add = List.map2 (+)

/// Logistic Sigmoid.
let logSigmoid x = (/) 1.0 ((+) 1.0 (exp -x))

/// Derivative of Logistic Sigmoid.
let deltaLogSigmoid x = (*) x ((-) 1.0 x)

/// Derivative of TanH i.e. sec^2h.
let deltaTanH x = (/) 1.0 <| (*) (cosh x) (cosh x)

/// Generate List of Random Elements.
let listRandElems count =
 let rec f (rand:System.Random) acc c = match c with | 0 -> acc | _ -> f rand <| rand.NextDouble()::acc <| (-) c 1
 f (System.Random()) List.empty count

/// Gradient. dFunc is the derivative of forward squashing function.
let gradient dFunc output target = (*) <| dFunc output <| (-) target output

/// Weighted Sum with Bias.
let weightedSum inputs weights bias = add bias <| List.map (dot inputs) weights

/// Delta or The Rate of Change.
let deltas learningRate gradients netOutputs = List.map <| smul learningRate <| mapToSecondList (*) gradients netOutputs



/// Represents a Network Layer.
type Layer = {
  Inputs: List<float>
  Weights: List<List<float>>
  Bias: List<float>
  Gradients: List<float>
  PrevDeltas: List<List<float>>
  BiasPrevDeltas: List<float>
  NetOutputs: List<float>
  }

/// Represents a Feed Forward Network.
type Network = {
 LearningRate: float
 Momentum: float
 Inputs: List<float>
 FirstHiddenLayer : Layer
 SecondHiddenLayer : Layer
 OutputLayer : Layer
 TargetOutputs: List<float>
 }

let feedForward net =
 let firstHiddenWeightedSum = weightedSum net.Inputs net.FirstHiddenLayer.Weights net.FirstHiddenLayer.Bias
 let firstHiddenNetOutputs = List.map tanh firstHiddenWeightedSum
 let secondHiddenWeightedSum = weightedSum firstHiddenNetOutputs net.SecondHiddenLayer.Weights net.SecondHiddenLayer.Bias
 let secondHiddenNetOutputs = List.map tanh secondHiddenWeightedSum
 let outputWeightedSum = weightedSum secondHiddenNetOutputs net.OutputLayer.Weights net.OutputLayer.Bias
 let outputs = List.map tanh outputWeightedSum
 {
  net with
   FirstHiddenLayer = {
                       net.FirstHiddenLayer with
                        Inputs = net.Inputs
                        NetOutputs = firstHiddenNetOutputs
                      }
   SecondHiddenLayer = {
                        net.SecondHiddenLayer with
                         Inputs = firstHiddenNetOutputs
                         NetOutputs = secondHiddenNetOutputs
                       }
   OutputLayer = {
                  net.OutputLayer with
                   Inputs = secondHiddenNetOutputs
                   NetOutputs = outputs
                 }
 }

let bpOutputLayer n m tOutputs (layer:Layer) =
 let grads = List.map2 (gradient deltaTanH) layer.NetOutputs tOutputs
 let bpDeltas = deltas n grads layer.Inputs
 let prevDeltasWithM = List.map (smul m) layer.PrevDeltas
 let newDeltas = List.map2 add bpDeltas prevDeltasWithM
 let weightsUpdate= List.map2 add layer.Weights newDeltas
 let biasDeltas = smul n grads
 let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
 let biasNewDeltas = add biasDeltas biasPrevDeltasWithM
 let biasUpdate = add layer.Bias biasNewDeltas
 {
  layer with
   Weights = weightsUpdate
   Bias = biasUpdate
   Gradients = grads
   PrevDeltas = newDeltas
   BiasPrevDeltas = biasNewDeltas
 }

let bpHiddenLayer n m layer nextLayer =
 let grads = mul (List.map deltaTanH layer.NetOutputs) (List.map (dot nextLayer.Gradients) (transpose nextLayer.Weights))
 let bpDeltas = deltas n grads layer.Inputs
 let prevDeltasWithM = List.map (smul m) layer.PrevDeltas
 let newDeltas = List.map2 add bpDeltas prevDeltasWithM
 let weightsUpdate = List.map2 add layer.Weights newDeltas
 let biasDeltas = smul n grads
 let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
 let biasNewDeltas = add biasDeltas biasPrevDeltasWithM
 let biasUpdate = add layer.Bias biasNewDeltas
 {
  layer with
   Weights = weightsUpdate
   Bias = biasUpdate
   Gradients = grads
   PrevDeltas = newDeltas
   BiasPrevDeltas = biasNewDeltas
 }

let backPropagate (net:Network) =
 let bpOutputLayer = bpOutputLayer net.LearningRate net.Momentum net.TargetOutputs net.OutputLayer
 let bpHidLayerWithHyperParams = bpHiddenLayer net.LearningRate net.Momentum
 let bpSecHidLayer = bpHidLayerWithHyperParams net.SecondHiddenLayer bpOutputLayer
 let bpFirstHidLayer = bpHidLayerWithHyperParams net.FirstHiddenLayer bpSecHidLayer
 {
  net with
   OutputLayer = bpOutputLayer
   SecondHiddenLayer = bpSecHidLayer
   FirstHiddenLayer =  bpFirstHidLayer
 }

(* Utility functions. *)
let splitToIO net = List.splitAt net.Inputs.Length
let validate net data =
 let inputs, targets = splitToIO net data
 { net with Inputs = inputs; TargetOutputs = targets } |> feedForward

/// Train Neural Net with epoch, kfold size, network and the data.
let rec train epoch kfold netAcc data_xs =
 let trainOnce net data =
  let inputs, targets = splitToIO net data
  { net with Inputs = inputs; TargetOutputs = targets } |> feedForward |> backPropagate
 let networkDistance network = distance network.TargetOutputs network.OutputLayer.NetOutputs
 let log path data = File.AppendAllText(path, data)
 let logToDataFile filename =
   let fullfilepath = @"C:\Users\Arya\Desktop\breast-cancer-cell\dat\"+filename
   log fullfilepath
 let errors trainedRms validatedRms = trainedRms + "," + validatedRms + "\n"
 let vectorToString (vector:List<float>) =
  let concatCommaSep (x:float) s = x.ToString("F6") + "," + s
  List.foldBack concatCommaSep vector ""
 let rec matrixToString (matrix:List<List<float>>) =
  let concatStringVector vector s = vectorToString vector + ";" + s
  List.foldBack concatStringVector matrix ""
 match epoch with
 | 0 -> netAcc
 | _ ->
  let shuffledData_xs = shuffle data_xs
  let trainSet, testSet = List.splitAt kfold shuffledData_xs
  let trained = List.fold trainOnce netAcc trainSet
  let trainedRms = networkDistance trained
  let validated = List.fold validate netAcc testSet
  let validatedRms = networkDistance validated

  if epoch % 100 = 0 then
   printfn "%f %f" trainedRms validatedRms
   (* write error *)
   logToDataFile "errors.dat" <| errors (trainedRms.ToString()) (validatedRms.ToString())
   (* write appropriate parameters.  -h1,h2,output weights and biases. *)
   let logNetworkParameters =
    (netAcc.FirstHiddenLayer.Weights |> matrixToString) + "," +
    (netAcc.FirstHiddenLayer.Bias |> vectorToString) + "," +
    (netAcc.SecondHiddenLayer.Weights |> matrixToString) + "," +
    (netAcc.SecondHiddenLayer.Bias |> vectorToString) + "," +
    (netAcc.OutputLayer.Weights |> matrixToString) + "," +
    (netAcc.OutputLayer.Bias |> vectorToString) + "\n"
   logToDataFile "weightsAndBiases.dat" <| logNetworkParameters
  train ((-) epoch 1) kfold trained shuffledData_xs

let inputSize = 9;
let hiddenSize = 8;
let outputSize = 2;

let network = {
 LearningRate = 0.001
 Momentum = 0.5
 Inputs = List.replicate inputSize 0.0
 FirstHiddenLayer = {
                     Inputs = List.empty
                     Weights = ((*) inputSize hiddenSize) |> listRandElems |> List.chunkBySize inputSize
                     Bias = listRandElems hiddenSize
                     Gradients = List.empty
                     PrevDeltas = List.replicate hiddenSize <| List.replicate inputSize 0.0
                     BiasPrevDeltas = List.replicate hiddenSize 0.0
                     NetOutputs = List.empty
 }
 SecondHiddenLayer = {
                      Inputs = List.empty
                      Weights = ((*) hiddenSize hiddenSize) |> listRandElems |> List.chunkBySize hiddenSize
                      Bias = listRandElems hiddenSize
                      Gradients = List.empty
                      PrevDeltas = List.replicate hiddenSize <| List.replicate hiddenSize 0.0
                      BiasPrevDeltas = List.replicate hiddenSize 0.0
                      NetOutputs = List.empty
 }
 OutputLayer = {
                Inputs = List.empty
                Weights = ((*) hiddenSize outputSize) |> listRandElems |> List.chunkBySize hiddenSize
                Bias = listRandElems outputSize
                Gradients = List.empty
                PrevDeltas = List.replicate outputSize <| List.replicate hiddenSize 0.0
                BiasPrevDeltas = List.replicate outputSize 0.0
                NetOutputs = List.empty
 }
 TargetOutputs = List.replicate outputSize 0.0
}

let dataToFloatList separator data = Regex.Split(data, separator) |> Array.map float |> Array.toList
let csvStrToFloatList = dataToFloatList ","
let loadData filename = (* replace with your current directory. *)
 File.ReadAllLines(@"C:\Users\Arya\Desktop\breast-cancer-cell\dataset\"+filename)
 |> Array.toList
 |> List.map csvStrToFloatList

let allData = loadData "data.csv"

let epoch = 1000
let kfold = 10
printfn "Training..."
let trainedNet = train epoch kfold network allData

let computeAccuracy network xss =
 let g size xs n =
  let isCorrect xs ys = if xs = ys then 1 else 0
  let heaveside x = if x <= 0.0 then -1.0 else 1.0
  let squeeze xs = List.map heaveside xs 
  let inputs, targets = List.splitAt size xs
  let netOutputs = (validate network inputs).OutputLayer.NetOutputs
  let squeezed = squeeze netOutputs
  (isCorrect squeezed targets) + n
 let correctItems = List.foldBack (g network.Inputs.Length) xss 0
 (float correctItems) / (float xss.Length)

printfn "Testing Model Accuracy: Crunching All Data..."
let accuracy = computeAccuracy trainedNet allData