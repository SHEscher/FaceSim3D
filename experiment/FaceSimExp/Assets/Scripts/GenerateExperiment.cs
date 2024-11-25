using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UXF;

public class GenerateExperiment : MonoBehaviour
{
    public GameObject[] heads_3D, headsT_3D;
    public static bool stim_3D; // stimulus type (will be set to 2D: false or 3D: true)
    public static bool training_block;

    int n_heads = 100;     // Number of 3D heads
    int n_theads = 10;     // Number of 3D training heads (4 training trials: 3 + 3 + 2 + 2 heads; where '2' are training catch-trials)
    private static int n_stimuli = 100;  // 30: pilot 1; 25: pilot 2; 100: main [Note: until 2023-01-06 this was n_stimuli = 25]
    public static int n_trialsblock = 60;  // trials per block
    public static Dictionary<string, object> blockDB = new Dictionary<string, object>(); // Current Session
    public static Dictionary<string, object> currentBlockDB = new Dictionary<string, object>(); // Current Block
    public static int n_blocks = 3; // 5 minutes each of 60 trials
    public static int n_training_trials = 4;

    public static List<Dictionary<string, object>> tripletBlockCombos;
    //static int trialIterator = 0;
    public static bool getDDB = false;
    // public static UXFDataTable tripletsDT;

    private bool one_time_debugger = true;

    // Variables for temporarily storing data requested from DDB
    private static string exclusiveStartKey;
    private static List<object> queryResponse;

    // Manually set experimental group
    private static string exp_group = "2D";  // "2D" or "3D"
    private static string ddbTable;

    private static int n_participants = 100;
    private static int n_totalTrials = n_stimuli * (n_stimuli - 1) * (n_stimuli - 2) / 6; // nC3 combinations
    public static int n_partiTrials = 6; // n_totalTrials/n_participants; // trials/block

    public static WebAWSDynamoDB webDB;

    // Initiates all the game object heads
    private void InitiateHeads(GameObject[] heads_3D, int n_heads)
    {
        for (int i = 0; i < n_heads; i++)
        {
            if (n_heads == n_theads)
            {
                heads_3D[i] = GameObject.FindGameObjectWithTag("HeadT" + (i + 1).ToString()); // training heads
            }
            else
            {
                heads_3D[i] = GameObject.FindGameObjectWithTag("Head" + (i + 1).ToString());
            }

            if (heads_3D[i] != null)
            {
                if (heads_3D[i].GetComponent<MeshRenderer>() == null)
                {
                    heads_3D[i].AddComponent<MeshRenderer>();
                }
                heads_3D[i].GetComponent<MeshRenderer>().material.SetFloat("_Metallic", 0f);
                heads_3D[i].GetComponent<MeshRenderer>().material.EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
                heads_3D[i].GetComponent<MeshRenderer>().material.EnableKeyword("_GLOSSYREFLECTIONS_OFF");
                heads_3D[i].GetComponent<MeshRenderer>().receiveShadows = false;
                heads_3D[i].GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                heads_3D[i].GetComponent<MeshRenderer>().enabled = false; // all heads initiated but not rendered
            }
        }
        return;
    }

    private void Start()
    {
        webDB = GameObject.FindGameObjectWithTag("WebDB").GetComponent<WebAWSDynamoDB>(); // AWS Script
        Dictionary<string, string> urlParams = URLParameters.GetSearchParameters();
        if (urlParams.ContainsKey("STUDY_ID"))
        {
            exp_group = urlParams["STUDY_ID"];
        }
        if (urlParams.ContainsKey("n_trialsblock"))
        {
            n_trialsblock = System.Int32.Parse(urlParams["n_trialsblock"]);
        }
        if (urlParams.ContainsKey("n_blocks"))
        {
            n_blocks = System.Int32.Parse(urlParams["n_blocks"]);
        }

        ddbTable = $"UXFData.FaceSim.TripletsIDB.{exp_group}";
        if (urlParams.ContainsKey("PROLIFIC_PID"))
        {
            if (urlParams["PROLIFIC_PID"].ToLower().Contains("debug"))
            {
                ddbTable += ".Pilot";
            }
        }
        else
        {
            ddbTable += ".Pilot";
        }
    }

    public void Generate(Session uxfSession)
    {
        StartCoroutine(CoGenerate(uxfSession));
        if (!Debug.isDebugBuild)
        {
            Cursor.visible = false;
        }
    }

    public IEnumerator CoGenerate(Session uxfSession)
    {
        if (uxfSession.ppid == "UnlockTriplets")
        {
            int i = 0;
            exclusiveStartKey = "";
            Debug.Log("Querying database for locked triplets...");
            while (i < 100)
            {
                Debug.Log("Querying...");
                webDB.QueryCustomBatchDataFromDB(ddbTable, "ppid_session_dataname", "000_s001_tripletsdb", HandleQueryData, "status", "L", "triplet_id, triplet", exclusiveStartKey);
                yield return new WaitUntil(() => getDDB);
                getDDB = false;

                UnLockTriplets();

                if (exclusiveStartKey == "none")
                {
                    Debug.Log("End reached.");
                    break;
                }
                i++;
            }
            Debug.Log("Finished unlocking triplets. Ending experiment.");
            uxfSession.End();
            yield break;
        }
        // Randomly assigning into 2D & 3D groups
        /*
        int randGroup = Random.Range(0, 100); //returns random number between 0-99
        if (randGroup<50){    stim_3D = false;    }
        else{    stim_3D = true;    }
        stim_3D = true; // Manually set to 2D
        */
        //webDB = GameObject.FindGameObjectWithTag("WebDB").GetComponent<WebAWSDynamoDB>(); // AWS Script
        uxfSession.participantDetails["SystemDateTime_StartExp"] = System.DateTime.Now.ToString("yyyy/MM/dd HH:mm:ss.fff");
        uxfSession.participantDetails["targetTripletDB"] = ddbTable;
        uxfSession.participantDetails["group_exp"] = exp_group;
        if (exp_group.CompareTo("2D") == 0)
        {
            stim_3D = false;
        }
        if (exp_group.CompareTo("3D") == 0)
        {
            stim_3D = true;
        }

        headsT_3D = new GameObject[n_theads];
        InitiateHeads(headsT_3D, n_theads);

        heads_3D = new GameObject[n_heads];
        InitiateHeads(heads_3D, n_heads);

        // Start training block
        training_block = true;
        IterateTrainingBlock(uxfSession);
        training_block = false;

        // Actual experiment
        CreateBlockXP(uxfSession, n_trialsblock);
    }

    private void IterateTrainingBlock(Session uxfSession)
    {
        List<Dictionary<string, object>> tripletCombos = getTrainingTriplets(n_training_trials); // get training triplets
        // Debug.Log(string.Join("--",tripletCombos));
        training_block = true;
        Block myBlock = uxfSession.CreateBlock(n_training_trials);
        IterateBlocks(myBlock, tripletCombos);
    }

    private void CreateBlockXP(Session uxfSession, int n_trialsblock)
    {
        //int PID = (int)uxfSession.participantDetails["parti_id"]; // pilot only
        int PID = -1;  // main experiment
        // Get from 100 separate blocks 1617 trials [??]
        List<Dictionary<string, object>> tripletBlockCombos = getTripletsForParticipant(PID); // 0-99 PID, pilot only
        for (int i = 0; i < tripletBlockCombos.Count(); i++)
        {
            blockDB.Add((string)tripletBlockCombos[i]["triplet"],
                        new Dictionary<string, object> {
                            {"triplet_id", (int)tripletBlockCombos[i]["triplet_id"]},
                            {"status", "U"}
                        }
            );
        }

        // if participant is a researcher: first reset TripletsDB
        if (uxfSession.ppid == "000")
        {
            // blockDB = resetAllTripletsDB(n_stimuli);
            UXFDataTable tripletsDT = new UXFDataTable(blockDB.Count(), new string[]{ "triplet", "status" });
            int tripletIterator = 0;

            for (int block_i = 1; block_i <= blockDB.Count() / n_trialsblock; block_i++)
            {
                Block testBlock = uxfSession.CreateBlock(n_trialsblock); // experimental blocks - 60 before break
                foreach (Trial trial in testBlock.trials)
                {
                    if (tripletIterator < tripletBlockCombos.Count)
                    {
                        // get the triplet head numbers for each trial
                        string[] headnum = ((string)tripletBlockCombos[tripletIterator]["triplet"]).Split('_');

                        // shuffling randomly for positioning the chosen triplets
                        int numOfInts = 3;
                        System.Random rand = new System.Random();
                        var ints = Enumerable.Range(0, numOfInts)
                                                .Select(i => new System.Tuple<int, int>(rand.Next(numOfInts), i))
                                                .OrderBy(i => i.Item1)
                                                .Select(i => i.Item2).ToArray();

                        int head1 = System.Int32.Parse(headnum[ints[0]]);
                        int head2 = System.Int32.Parse(headnum[ints[1]]);
                        int head3 = System.Int32.Parse(headnum[ints[2]]);

                        trial.settings.SetValue("head1", head1);
                        trial.settings.SetValue("head2", head2);
                        trial.settings.SetValue("head3", head3);

                        //Session.instance.settings.UpdateWithDict(new Dictionary<string, object> { { tripletBlockCombos[tripletIterator], "U" } });
                        UXFDataRow trialRow = new UXFDataRow(){ ("triplet", (string)tripletBlockCombos[tripletIterator]["triplet"]), ("status", "U") };
                        tripletsDT.AddCompleteRow(trialRow);
                        tripletIterator++;
                    }
                    // Debug.Log("trial number: " + (trial.number - 4).ToString() + "/" + (Session.instance.LastTrial.number - 4).ToString());
                }
            }
            Session.instance.SaveDataTable(tripletsDT,"TripletsDB"); // File Writing TripletsDB
            // webDB.HandleDataTable(tripletsDT, "FaceSim", "000", 1,"trial_results", UXFDataType.TrialResults,0);
        }
        else // get tripletsDB from DDB online
        {
            Debug.Log("Session details: last trial" + uxfSession.LastTrial.block.number.ToString() + ", " + uxfSession.LastTrial.number.ToString());
            Debug.Log("Session details: first trial" + uxfSession.FirstTrial.block.number.ToString() + ", " + uxfSession.FirstTrial.number.ToString());
            currentBlockDB = getTripletsDB(currentBlockDB, uxfSession, tripletBlockCombos);

            /* // Get trials from participants
            int tripletIterator = 0;
            Block testBlock = uxfSession.CreateBlock(n_trialsblock); // experimental blocks - 60 before break
            //
            int PID = (int)uxfSession.participantDetails["parti_id"]; // pilot only
            List<string> tripletBlockCombos = getTripletsForParticipant(PID); // all triplets of n stimuli
            */
        }
    }

    private void IterateBlocks(Block testBlock, List<Dictionary<string, object>> tripletCombos)
    {
        int tripletIterator = 0;

        // Debug.Log("Total Triplets: " + tripletBlockCombos.Count);
        foreach (Trial trial in testBlock.trials)
        {
            if (tripletIterator < tripletCombos.Count)
            {
                // set inter-trial time
                float intertrial = Random.Range(0.4f, 0.6f);
                trial.settings.SetValue("crossTrialTime", intertrial); // 0.5 seconds

                // get the triplet head numbers for each trial
                string[] headnum = ((string)tripletCombos[tripletIterator]["triplet"]).Split('_');

                // shuffling randomly for positioning the chosen triplets
                int numOfInts = 3;
                System.Random rand = new System.Random();
                var ints = Enumerable.Range(0, numOfInts)
                                        .Select(i => new System.Tuple<int, int>(rand.Next(numOfInts), i))
                                        .OrderBy(i => i.Item1)
                                        .Select(i => i.Item2).ToArray();

                int head1 = System.Int32.Parse(headnum[ints[0]]);
                int head2 = System.Int32.Parse(headnum[ints[1]]);
                int head3 = System.Int32.Parse(headnum[ints[2]]);

                // Debug.Log(string.Join("--", headnum));
                // Debug.Log(head1.ToString() + " " + head2.ToString() + " " + head3.ToString());

                trial.settings.SetValue("triplet", (string)((Dictionary<string, object>)tripletCombos[tripletIterator])["triplet"]);
                trial.settings.SetValue("triplet_id", (int)((Dictionary<string, object>)tripletCombos[tripletIterator])["triplet_id"]);
                trial.settings.SetValue("head1", head1);
                trial.settings.SetValue("head2", head2);
                trial.settings.SetValue("head3", head3);
                tripletIterator++;
            }
            // example trial set from AddingTask trial.settings.SetValue("number 1", 2);
        }
    }

    // Base Triplets Combo Data
    private static Dictionary<string, object> resetAllTripletsDB(int n_stimuli)
    {
        int seed = 42; // consistent randomization

        List<string> shuffledTriplets = new List<string>(getAllTriplets(n_stimuli)); //161,700 combinations in main experiment (n_totalTrials)
        Shuffle(shuffledTriplets, seed);
        // Printing all tripletCombos        // Debug.Log(string.Join("--", shuffledTriplets));

        Dictionary<string, object> triplet_dict = new Dictionary<string, object>();
        for (int i = 0; i < shuffledTriplets.Count(); i++) {
            triplet_dict.Add(shuffledTriplets[i], "U");
        }
        return triplet_dict;
    }

    public static bool setTripletsDB(int triplet_id, string ppid_session_dataname, string status, string triplet)
    {
        webDB.PutCustomDataInDB(ddbTable, new Dictionary<string, object>(){
            { "ppid_session_dataname", ppid_session_dataname },
            { "triplet_id", triplet_id },
            { "status", status },
            { "triplet", triplet }});
        return true;
    }

    public Dictionary<string, object> getTripletsDB(Dictionary<string, object> currentBlockDB, Session uxfSession, List<Dictionary<string, object>> tripletBlockCombos)
    {
        Debug.Log("Running coroutine GetCurrentBlockDB");
        //StartCoroutine(RunEndCoroutine(webAWSDynamoDB, currentBlockDB, uxfSession, tripletBlockCombos));
        StartCoroutine(GetCurrentBlockDB(currentBlockDB, uxfSession, tripletBlockCombos)); // Coroutines help in waiting while triplets are being fetched

        return currentBlockDB;
    }

    private IEnumerator RunEndCoroutine(Dictionary<string, object> currentBlockDB, Session uxfSession, List<Dictionary<string, object>> tripletBlockCombos)
    {
        yield return StartCoroutine(GetCurrentBlockDB(currentBlockDB, uxfSession, tripletBlockCombos));
    }

    IEnumerator GetCurrentBlockDB(Dictionary<string, object> currentBlockDB, Session uxfSession, List<Dictionary<string, object>> tripletBlockCombos)
    {
        if (n_blocks > 0)
        {
            // int cblock_count = currentBlockDB.Count();
            /* TO BE UPDATED TO BATCH GET

            // Using DDB_GetItem
            while (currentBlockDB.Count() <= 59) // Add < 60 U
            {
                webDB.GetCustomDataFromDB("UXFData.FaceSim.Triplets",
                                            "triplet", tripletBlockCombos[trialIterator],
                                            HandleMyData,
                                            "ppid_session_dataname", "000_s001_tripletsdb");
                yield return new WaitUntil(() => getDDB);
                getDDB = false;
                Debug.Log("Current block count after: " + currentBlockDB.Count.ToString());
                trialIterator++;

                if (trialIterator == tripletBlockCombos.Count() - 1) // resetting trialIterator to 0 after one full iteration of session trials
                {
                    trialIterator = 0;
                }
            }
            */

            Debug.Log("Trying to get batch triplets");

            if(one_time_debugger)
            {
                exclusiveStartKey = "";
                //exclusiveStartKey = "{\"ppid_session_dataname\": {\"S\": \"000_s001_tripletsdb\"},\"triplet_id\": {\"N\": \"161699\"}}";
                int attempt = 1;
                while (currentBlockDB.Count() < n_trialsblock * n_blocks) // Add < 60 * 3 U
                {
                    Debug.Log($"Query attempt: {attempt++}");
                    webDB.QueryCustomBatchDataFromDB(ddbTable, "ppid_session_dataname", "000_s001_tripletsdb", HandleQueryData, "status", "U", "triplet_id, triplet", exclusiveStartKey);

                    yield return new WaitUntil(() => getDDB);
                    getDDB = false;

                    queryResponse.Shuffle();
                    yield return LockTriplets();
                    Debug.Log("Current total trial count after most recent query: " + currentBlockDB.Count.ToString());


                    if (exclusiveStartKey == "none")
                    {
                        Debug.Log("No more available triplets in database!");
                        break;
                    }
                }
                one_time_debugger = false;
            }

            Debug.Log("Current total trial count: " + currentBlockDB.Count().ToString());
            Debug.Log(string.Join("\n", currentBlockDB.Keys.ToList()));

            List<string> remainingTrialsTriplets = currentBlockDB.Keys.ToList();
            List<Dictionary<string, object>> remainingTrials = new List<Dictionary<string, object>>();
            foreach (string triplet in remainingTrialsTriplets)
            {
                remainingTrials.Add(
                    new Dictionary<string, object>
                    {
                        {"triplet", triplet},
                        {"triplet_id", (int)((Dictionary<string, object>)currentBlockDB[triplet])["triplet_id"]}
                    }
                );
            }

            for (int block_i = 1; block_i <= n_blocks; block_i++)
            {
                Debug.Log("Remaining trials count: " + remainingTrials.Count().ToString());

                if (remainingTrials.Any()) // all triplets covered or 20 minutes of triplets or 4 blocks of session complete !!!!! Remaining FALL THROUGH
                {
                    Debug.Log("TestBlock number: " + block_i.ToString());
                    Block testBlock;
                    if (remainingTrials.Count() >= n_trialsblock)
                    {
                        testBlock = uxfSession.CreateBlock(n_trialsblock); // experimental blocks - 60 before break
                        IterateBlocks(testBlock, remainingTrials);
                        remainingTrials = remainingTrials.Skip(n_trialsblock).ToList();
                    }
                    else
                    {
                        testBlock = uxfSession.CreateBlock(remainingTrials.Count()); // experimental blocks - 60 before breaK
                        IterateBlocks(testBlock, remainingTrials);
                        remainingTrials.Clear();
                    }
                    Debug.Log("Trials added to block: " + testBlock.trials.Count().ToString());
                    //n_blocks--;
                    // currentBlockDB = getTripletsDB(currentBlockDB, uxfSession, tripletBlockCombos);
                }
            }
        }
        yield return null;
    }

    private static List<Dictionary<string, object>> getTrainingTriplets(int n)
    {
        List<int> numberList = new List<int>();
        for (int i = 1; i <= 10; i++) // 10 training heads available (n_theads)
        {
            numberList.Add(i);
        }
        // Random selection out of tripletCombos
        // Debug.Log(string.Join("-", pairCombos.OrderBy(x => rnd.Next(pairCombos.Count)).Take(1)));
        List<string> trainTriplets = getTriplets(numberList);

        List<Dictionary<string, object>> trainTripsDict = new List<Dictionary<string, object>>();
        int triplet_id = 1;
        foreach (string triplet in trainTriplets)
        {
            trainTripsDict.Add(
                new Dictionary<string, object>{
                    {"triplet", triplet},
                    {"triplet_id", triplet_id}
                }
            );
            triplet_id++;
        }

        System.Random rand = new System.Random();
        return new List<Dictionary<string, object>>(trainTripsDict.OrderBy(x => rand.Next()).Take(n));
    }

    private static List<string> getAllTriplets(int n)
    {
        //int seed = 52471331;
        int[] fHeads = Enumerable.Range(1, 50).ToArray();
        int[] mHeads = Enumerable.Range(51, 50).ToArray();

        fHeads.Shuffle(); //Shuffle(fHeads, seed);
        mHeads.Shuffle(); //Shuffle(mHeads, seed);

        List<int> numberList = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (i%2 == 0)
            {
                numberList.Add(fHeads[i/2]); // Add females
            }
            else
            {
                numberList.Add(mHeads[(i-1)/2]); // Add males
            }
        }

        // Random selection out of tripletCombos
        // Debug.Log(string.Join("-", pairCombos.OrderBy(x => rnd.Next(pairCombos.Count)).Take(1)));
        return getTriplets(numberList);
    }

    // Gets the triplet Combinations from a number list
    private static List<string> getTriplets(List<int> items)
    {
        var combinations =
            from a in items
            from b in items
            from c in items
            where a < b && b < c
            orderby a, b, c
            select new { A = a, B = b, C = c };

        List<string> tripletCombos = new List<string>();

        foreach (var triplet in combinations)
            tripletCombos.Add(triplet.A + "_" + triplet.B + "_" + triplet.C);

        return tripletCombos;
    }

    public static void Shuffle<T>(IList<T> list, int seed)
    {
        var rng = new System.Random(seed);
        int n = list.Count;

        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }

    // First pilot: manual PIDs and online pilot binning
    private static List<Dictionary<string, object>> getTripletsForParticipant(int n)
    {
        int seed = 42; // consistent randomization

        List<string> shuffledTriplets = new List<string>(getAllTriplets(n_stimuli)); // triplet combinations (len := n_totalTrials)
        // [2023-01-06: Note that n_stimuli was changed to 100!]
        Shuffle(shuffledTriplets, seed);

        List<Dictionary<string, object>> shuffledDictTrips = new List<Dictionary<string, object>>();
        int comboNumber = 1;
        foreach (string triplet in shuffledTriplets)
        {
            shuffledDictTrips.Add(
                new Dictionary<string, object>
                {
                    {"triplet_id", comboNumber},
                    {"triplet", triplet}
                }
            );
            comboNumber++;
        }

        if (n < 0)  // for main experiment
        {
            return shuffledDictTrips;
        }
        else  // should only be triggered for pilot experiment
        {
            int nTotalCombos = shuffledDictTrips.Count;

            // Division of the full set among participants
            int nPartiCombos = nTotalCombos / n_participants; // Here number of participants is 100 for main experiment

            //secondFiveItems = myList.Skip(5).Take(5);
            return shuffledDictTrips.Skip(n * nPartiCombos).Take(nPartiCombos).ToList();
        }
    }


    public static void HandleGetTriplet(Dictionary<string, object> item)
    {
        // Define target status: will only change items that match this status
        string targetStatus = "U"; //Session.instance.ppid == "UnlockTriplets" ? "L" : "U";

        // Items whose statuses match the target will be changed to this new status
        string newStatus    = "L"; //Session.instance.ppid == "UnlockTriplets" ? "U" : "L";

        // Values from GetItem
        string triplet = item["triplet"].ToString();
        string status = item["status"].ToString();
        int triplet_id = System.Convert.ToInt32(item["triplet_id"]);

        if (status == targetStatus)
        {
            // Lock triplet
            setTripletsDB(triplet_id, "000_s001_tripletsdb", newStatus, triplet);

            // Save triplet to current experiment
            currentBlockDB.Add(triplet,
                new Dictionary<string, object>{
                    {"triplet_id", triplet_id},
                    {"status", newStatus}
                }
            );
        }

        getDDB = true;
    }

    public static void HandleQueryData(Dictionary<string, object> response)
    {
        queryResponse = (List<object>)response["jsonResults"];
        Debug.Log($"Query response items count: {queryResponse.Count()}");
        exclusiveStartKey = (string)response["lastEvaluatedKey"];

        getDDB = true;
    }

    public IEnumerator LockTriplets()
    {
        foreach (object dictriplets in queryResponse)
        {
            if (Session.instance.ppid != "UnlockTriplets" && currentBlockDB.Count() >= n_trialsblock * n_blocks) { break; }
            Dictionary<string, object> itriplet = (Dictionary<string, object>)dictriplets;
            object tripletValue; object triplet_idValue;
            //bool hasValue = itriplet.TryGetValue("triplet", out tripletValue);
            if (itriplet.TryGetValue("triplet", out tripletValue) && itriplet.TryGetValue("triplet_id", out triplet_idValue))
            {
                string triplet = (string) tripletValue;
                int triplet_id = System.Convert.ToInt32(triplet_idValue);

                // Double check status before locking triplet
                webDB.GetCustomDataFromDB(ddbTable, "ppid_session_dataname", "000_s001_tripletsdb", HandleGetTriplet, "triplet_id", triplet_id);
                yield return new WaitUntil(() => getDDB);
                getDDB = false;
            }
            else
            {
                Debug.Log("Key not present");
            }
        }
    }

    public void UnLockTriplets()
    {
        foreach (object dictriplets in queryResponse)
        {
            Dictionary<string, object> itriplet = (Dictionary<string, object>)dictriplets;
            object tripletValue; object triplet_idValue;
            //bool hasValue = itriplet.TryGetValue("triplet", out tripletValue);
            if (itriplet.TryGetValue("triplet", out tripletValue) && itriplet.TryGetValue("triplet_id", out triplet_idValue))
            {
                string triplet = (string) tripletValue;
                int triplet_id = System.Convert.ToInt32(triplet_idValue);

                // Unlock triplets
                setTripletsDB(triplet_id, "000_s001_tripletsdb", "U", triplet);
            }
            else
            {
                Debug.Log("Key not present");
            }
        }
    }

    public void SaveCurrentTrialResults()
    {
        // Because this is called at the end of the trial, corresponding data is actually found in PrevTrial
        Session.instance.CurrentTrial.SaveJSONSerializableObject(Session.instance.PrevTrial.result.GetBaseDict, "trial_results", UXFDataType.OtherTrialData);
    }

    public void SaveParticipantDetails()
    {
        Session.instance.SaveJSONSerializableObject(Session.instance.participantDetails, "participant_details", UXFDataType.OtherSessionData);
    }
}
