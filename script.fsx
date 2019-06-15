(* drop out *)

open System.IO
open System.Threading.Tasks
open System.Text.RegularExpressions

type Scalar = Scalar of float
type Vector = Vector of float list

/// Matrix Transpose.
let transpose xss =
    let rec f xss acc =
        let b = List.exists (List.isEmpty) xss
        if b then List.rev acc
        else
            let xs  = List.map List.head xss
            let yss = List.map (List.skip 1) xss
            f yss (xs::acc)
    if xss = [] then [] else f xss []

/// Vector Multiplication.
let mul = List.map2 (*)

let min = List.map2 (-)

let square x = x * x

/// The Dot Product of xs and ys.
let dot xs ys = mul xs ys |> List.sum


let distance xs ys =
  min xs ys
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


/// Maps the elements the first list (xs),
///    to second list (ys) using the mapper function.
let mapToSecondList f xs ys =
    let mapEach x = List.map (f x) ys
    List.map mapEach xs

/// Scalar Vector Multiplication.
let smul c = List.map ((*) c)

/// Vector Addition.
let add = List.map2 (+)

/// Logistic Sigmoid.
let logSigmoid x = 1.0 / (1.0 + (exp (-x)))

/// Derivative of Logistic Sigmoid.
let deltaLogSigmoid x = x * (1.0 - x)

/// Derivative of TanH i.e. sec^2h.
let deltaTanH x = 1.0  / square (cosh x)

/// Generate List of Random Elements.
let rxs count =
    let rand s () = System.Random(s).NextDouble()
    let rands s   = rand s
    let seededrxs = List.map rands [1..count]
    List.map (fun r -> r()) seededrxs

/// Gradient. f is the derivative of forward squashing function.
let gradient f output target = (f output) * (target - output)

/// Weighted Sum with Bias.
let weightedSum inputs weights bias = List.map (dot inputs) weights |> add bias

/// Delta or The Rate of Change.
let deltas learningRate gradients netOutputs =
    List.map (smul learningRate) (mapToSecondList (*) gradients netOutputs)

// Represents a Network Size.
type Size =
 {
     Input : int
     Hidden: int
     Output: int
 }

/// Represents a Network Layer.
type Layer = {
    Inputs        : float list
    Weights       : float list list
    Bias          : float list
    Gradients     : float list
    PrevDeltas    : float list list
    BiasPrevDeltas: float list
    NetOutputs    : float list
    }

/// Represents a Feed Forward Network.
type Network = {
    LearningRate      : float
    Momentum          : float
    Size              : Size
    Inputs            : float list
    FirstHiddenLayer  : Layer
    SecondHiddenLayer : Layer
    OutputLayer       : Layer
    TargetOutputs     : float list
    }

let feedForward net =

 let firstHiddenWeightedSum  = weightedSum
                                          net.Inputs net.FirstHiddenLayer.Weights
                                          net.FirstHiddenLayer.Bias
 let firstHiddenNetOutputs   = List.map tanh firstHiddenWeightedSum
 let secondHiddenWeightedSum = weightedSum
                                          firstHiddenNetOutputs
                                          net.SecondHiddenLayer.Weights
                                          net.SecondHiddenLayer.Bias
 let secondHiddenNetOutputs  = List.map tanh secondHiddenWeightedSum
 let outputWeightedSum       = weightedSum
                                          secondHiddenNetOutputs
                                          net.OutputLayer.Weights
                                          net.OutputLayer.Bias
 let outputs                 = List.map tanh outputWeightedSum
 
 { net with
     FirstHiddenLayer =
      {
          net.FirstHiddenLayer with
              Inputs     = net.Inputs
              NetOutputs = firstHiddenNetOutputs
      }
     SecondHiddenLayer =
      {
          net.SecondHiddenLayer with
             Inputs     = firstHiddenNetOutputs
             NetOutputs = secondHiddenNetOutputs
      }
     OutputLayer =
      {
         net.OutputLayer with
             Inputs     = secondHiddenNetOutputs
             NetOutputs = outputs
      }
  }


let bpOutputLayer n m tOutputs (layer:Layer) =
    let grads               = List.map2 (gradient deltaTanH) layer.NetOutputs tOutputs
    let bpDeltas            = deltas n grads layer.Inputs
    let prevDeltasWithM     = List.map (smul m) layer.PrevDeltas
    let newDeltas           = List.map2 add bpDeltas prevDeltasWithM
    let weightsUpdate       = List.map2 add layer.Weights newDeltas
    let biasDeltas          = smul n grads
    let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
    let biasNewDeltas       = add biasDeltas biasPrevDeltasWithM
    let biasUpdate          = add layer.Bias biasNewDeltas
    {
        layer with
                  Weights        = weightsUpdate
                  Bias           = biasUpdate
                  Gradients      = grads
                  PrevDeltas     = newDeltas
                  BiasPrevDeltas = biasNewDeltas
    }

let bpHiddenLayer n m layer nextLayer =
    let grads               = mul (List.map deltaTanH layer.NetOutputs)
                                  (List.map
                                      (dot nextLayer.Gradients)
                                      (transpose nextLayer.Weights))
    let bpDeltas            = deltas n grads (layer.Inputs)
    let prevDeltasWithM     = List.map (smul m) layer.PrevDeltas
    let newDeltas           = List.map2 add bpDeltas prevDeltasWithM
    let weightsUpdate       = List.map2 add layer.Weights newDeltas
    let biasDeltas          = smul n grads
    let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
    let biasNewDeltas       = add biasDeltas biasPrevDeltasWithM
    let biasUpdate          = add layer.Bias biasNewDeltas
 
    {
        layer with
                  Weights        = weightsUpdate
                  Bias           = biasUpdate
                  Gradients      = grads
                  PrevDeltas     = newDeltas
                  BiasPrevDeltas = biasNewDeltas
    }

let backPropagate net =
    
    let bpOutputLayer             = bpOutputLayer
                                        net.LearningRate net.Momentum
                                        net.TargetOutputs net.OutputLayer
    let bpHidLayerWithHyperParams = bpHiddenLayer
                                        net.LearningRate
                                        net.Momentum
    let bpSecHidLayer             = bpHidLayerWithHyperParams
                                        net.SecondHiddenLayer
                                        bpOutputLayer
    let bpFirstHidLayer           = bpHidLayerWithHyperParams
                                        net.FirstHiddenLayer
                                        bpSecHidLayer
    {
        net with
                OutputLayer       = bpOutputLayer
                SecondHiddenLayer = bpSecHidLayer
                FirstHiddenLayer  = bpFirstHidLayer
    }

(* Utility functions. *)
let splitToIO net = List.splitAt net.Inputs.Length

let validate net data =
    let inputs, targets = splitToIO net data
    feedForward { net with Inputs = inputs; TargetOutputs = targets }

let vectorToString (vector:List<float>) =
     let concatCommaSep (x:float) s =
         if s = "" then x.ToString("F6")
         else x.ToString("F6") + "," + s
     List.foldBack concatCommaSep vector ""

let matrixToString  matrix =
     let concatStringVector vector s = vectorToString vector + ";" + s
     List.foldBack concatStringVector matrix ""

let relativePath = @"D:\Projects\AI\breast-cancer-cell"
let datPath relativePath =
    let dat = @"\dat\"
    relativePath + dat

let networkDistance network =
    distance network.TargetOutputs network.OutputLayer.NetOutputs

let log path data = File.AppendAllText(path, data)

let logToDataFile filename =
     let filenamePath = @"\" + filename
     let fullfilepath = datPath relativePath + filenamePath
     log fullfilepath

let logBest epoch trainedRms validatedRms errors netAcc =

    if epoch % 100 = 0 then
            printfn "%f %f" trainedRms validatedRms
            
            (* write error *)
            logToDataFile "errors.dat" <| errors (trainedRms.ToString()) (validatedRms.ToString())
            
            (* write appropriate parameters.  -h1,h2,output weights and biases. *)
            let logNetworkParameters =
                                  (netAcc.FirstHiddenLayer.Weights  |> matrixToString) + "|" +
                                  (netAcc.FirstHiddenLayer.Bias     |> vectorToString) + "|" +
                                  (netAcc.SecondHiddenLayer.Weights |> matrixToString) + "|" +
                                  (netAcc.SecondHiddenLayer.Bias    |> vectorToString) + "|" +
                                  (netAcc.OutputLayer.Weights       |> matrixToString) + "|" +
                                  (netAcc.OutputLayer.Bias          |> vectorToString) + "\n"

            logToDataFile "weightsAndBiases.dat" logNetworkParameters
    else ()

let trainOnce net data =
        let inputs, targets = splitToIO net data
        { net with Inputs = inputs; TargetOutputs = targets }
        |> feedForward
        |> backPropagate

let errors trainedRms validatedRms = trainedRms + "," + validatedRms + "\n"

/// Train Neural Net with epoch, kfold size, network and the data.
let rec train epoch kfold netAcc data_xs =
    match epoch with
    | 0 -> netAcc
    | _ ->
        let shuffledData_xs   = shuffle data_xs
        let trainSet, testSet = List.splitAt kfold shuffledData_xs
        let trained           = lazy List.fold trainOnce netAcc trainSet
        let trained'          = Task.Factory
                                    .StartNew<Network>(fun () -> trained.Value)
                                    .Result
        let trainedRms        = networkDistance trained'
        let validated         = lazy List.fold validate netAcc testSet
        let validated'        = Task.Factory
                                    .StartNew<Network>(fun () -> validated.Value)
                                    .Result
        let validatedRms      = networkDistance validated'
        logBest epoch trainedRms validatedRms errors netAcc
        train ((-) epoch 1) kfold trained' shuffledData_xs


let size = { Input = 9; Hidden = 8; Output = 2 }

let network = {
    LearningRate     = 0.001
    Momentum         = 0.5
    Size             = size
    Inputs           = List.replicate size.Input 0.0
    FirstHiddenLayer =
     {
         Inputs         = []
         Weights        = ((*) size.Input size.Hidden)
                          |> rxs
                          |> List.chunkBySize size.Input
         Bias           = rxs size.Hidden
         Gradients      = []
         PrevDeltas     = List.replicate size.Hidden <| List.replicate size.Input 0.0
         BiasPrevDeltas = List.replicate size.Hidden 0.0
         NetOutputs     = []
     }
    SecondHiddenLayer =
     {
         Inputs         = []
         Weights        = ((*) size.Hidden size.Hidden)
                          |> rxs
                          |> List.chunkBySize size.Hidden
         Bias           = rxs size.Hidden
         Gradients      = []
         PrevDeltas     = List.replicate size.Hidden <| List.replicate size.Hidden 0.0
         BiasPrevDeltas = List.replicate size.Hidden 0.0
         NetOutputs     = []
     }
    OutputLayer =
     {
         Inputs         = []
         Weights        = ((*) size.Hidden size.Output)
                          |> rxs
                          |> List.chunkBySize size.Hidden
         Bias           = rxs size.Output
         Gradients      = []
         PrevDeltas     = List.replicate size.Output <| List.replicate size.Hidden 0.0
         BiasPrevDeltas = List.replicate size.Output 0.0
         NetOutputs     = []
     }
    TargetOutputs       = List.replicate size.Output 0.0
   }

let dataToFloatList separator data =
    Regex.Split(data, separator)
    |> Array.map float
    |> Array.toList

let csvStrToFloatList = dataToFloatList ","
let loadData filepath =
    if File.Exists(filepath)
    then Some (File.ReadAllLines filepath
               |> Array.toList
               |> List.map csvStrToFloatList) //*** check data format
    else None

let allData =
    let dataFilePath = @"\dataset\data.csv"
    loadData (relativePath + dataFilePath)

let epoch = 1000
let kfold = 10

let trainedNet = Option.map <| train epoch kfold network

let computeAccuracy network xss =
    let g size xs n =
        let isCorrect x y   = if x = y    then  1   else 0
        let heaveside x     = if x <= 0.0 then -1.0 else 1.0
        let squeeze xs      = List.map heaveside xs 
        let inputs, targets = List.splitAt size xs
        let netOutputs      = (validate network inputs).OutputLayer.NetOutputs
        let squeezed        = squeeze netOutputs
        (isCorrect squeezed targets) + n
    let correctItems = List.foldBack (g network.Inputs.Length) xss 0
    (float correctItems) / (float xss.Length)

let flip f x y = f y x
let accuracy trainedNet allData =  Option.bind (fun data -> Option.map (flip computeAccuracy data) trainedNet) allData
                                |> Option.fold (fun a x -> a + x) 0.0

printfn "Training..."
let trained = trainedNet allData

printfn "Testing Model Accuracy: Crunching All Data..."
let modelAccuracy = accuracy trained allData
