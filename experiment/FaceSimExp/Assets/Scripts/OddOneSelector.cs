using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;
using UXF;
using System.Runtime.InteropServices;

public class OddOneSelector : MonoBehaviour
{
#if UNITY_WEBGL
    [DllImport("__Internal")]
    private static extern void OpenURL(string url);
#endif

    // clicks and button recorders
    public static bool clicked1, clicked2, clicked3, nextTrial;

    // setting trial booleans
    // experiment exit 
    [HideInInspector]
    public bool exit_exp = false;

    // Logger counters
    int one_time_logger = 1;

    // time-holders for stimulus response start and end
    public float t_stim, t_end;

    // Colors
    private Color defaultColor = new Color(1f, 1f, 1f, 0.0f);
    public Color activeColor;

    // current time counters
    public static float t_curr, t_curr0, t_break, t_resp;
    public float maxt_break = 30f; // Maximum breaktime

    // mouse or keyboard
    public bool mouse;

    // catch trial variables
    bool catch_trial = false;
    int catch_head = 0;
    int catch_count = 0;
    bool caught = false;
    static IEnumerable<int> catch_sequence = Enumerable.Range(6, GenerateExperiment.n_trialsblock - 5).OrderBy(n => n * n * (new System.Random(3)).Next());
    List<int> catch_random3 = catch_sequence.Distinct().Take(3).ToList();

    // Game Objects On Trial Scene
    // public Camera camera;

    private GameObject camera1, camera2, camera3;

    // feedback text
    private GameObject text_feed;

    // end training feedback;
    private GameObject end_training, end_block;

    // reference cross
    private GameObject ref_cross;

    // correct response
    public static int correct_resp;

    // heads and head selected
    public static int head1, head2, head3, head_selected;

    // stimulus heads
    private GameObject head1_3D, head2_3D, head3_3D;

    // object hierarchy maintainers
    private GameObject heads3D;

    // background planes
    private GameObject plane1, plane2, plane3;

    // stimulus-response indicator // commenting
    private GameObject stim_indicator;

    // keyboard response keys
    private KeyCode[] lFace = {KeyCode.LeftArrow};
    private KeyCode[] mFace = {KeyCode.DownArrow, KeyCode.UpArrow};
    private KeyCode[] rFace = {KeyCode.RightArrow};

    // rotator targets
    private GameObject r_sphere1, r_sphere2, r_sphere3;
    private GameObject f_sphere1, f_sphere2, f_sphere3;
    private Vector3 rotateDir = new Vector3(0, 0, 1);
    public float rot_time;

//  int tripletIterator = 0;

    // Funtion starting with the OnTrialBegin
    public void RunOddOneSelectorTrial(Trial trial)
    {
        trial.result["SystemDateTime_BeginTrial"] = System.DateTime.Now.ToString("yyyy/MM/dd HH:mm:ss.fff");
        //trial.result["keyPress"] = "";

        // Frame-rate
        Application.targetFrameRate = 30;
        QualitySettings.vSyncCount = 0;

        // Camera variables
        camera1 = GameObject.FindGameObjectWithTag("Camera1");
        camera2 = GameObject.FindGameObjectWithTag("Camera2");
        camera3 = GameObject.FindGameObjectWithTag("Camera3");

        heads3D = GameObject.FindGameObjectWithTag("Heads3D");

        // trial variables
        // float t_crosstrial = trial.settings.GetFloat("crossTrialTime");
        head1 = trial.settings.GetInt("head1");
        head2 = trial.settings.GetInt("head2");
        head3 = trial.settings.GetInt("head3");

        // local storage variables for UI and time control
        clicked1 = false;
        clicked2 = false;
        clicked3 = false;
        nextTrial = false;
        t_curr = 0f;

        // logger variables
        t_resp = 0f;
        correct_resp = 0;
        head_selected = 0;
        t_break = 0f;

        if (trial.number <= GenerateExperiment.n_training_trials)
        {
            // 3D training head initializations
            head1_3D = GameObject.FindGameObjectWithTag("HeadT" + head1.ToString());
            head2_3D = GameObject.FindGameObjectWithTag("HeadT" + head2.ToString());
            head3_3D = GameObject.FindGameObjectWithTag("HeadT" + head3.ToString());

            // Introducing catch trials in train & test trials

            if (trial.number >= 3)
            {
                catch_trial = true;
                catchTheTrial(trial);
            }
            else
            {
                catch_trial = false;
            }
        }
        else
        {
            // 3D head initializations
            head1_3D = GameObject.FindGameObjectWithTag("Head" + head1.ToString());
            head2_3D = GameObject.FindGameObjectWithTag("Head" + head2.ToString());
            head3_3D = GameObject.FindGameObjectWithTag("Head" + head3.ToString());

            //Debug.Log(trial.number); Debug.Log(System.String.Join(" - ", sequence));Debug.Log(System.String.Join(" - ", random3)); Debug.Log((trial.number - 4) % (GenerateExperiment.n_partiTrials / 3));Debug.Log(catch_random);
            if ((trial.number - GenerateExperiment.n_training_trials) % (GenerateExperiment.n_trialsblock) == 0)
            {
                if (catch_random3.Contains(GenerateExperiment.n_trialsblock))
                {
                    catch_trial = true;
                }
            }
            else if (catch_random3.Contains((trial.number - GenerateExperiment.n_training_trials) % (GenerateExperiment.n_trialsblock)))
            {
                catch_trial = true;
            }
            else
            {
                // Debug.Log("False");
                catch_trial = false;
            }

            if (catch_trial)
            {
                catchTheTrial(trial);
            }
            // Debug.Log("Catch Trial: " + catch_trial.ToString());
        }

        if (Session.instance.ppid == "000" && trial.number > GenerateExperiment.n_training_trials)
        {

            // Game Object UIs
            camera2.transform.position = new Vector3(960, 539.7f, -30);
            ref_cross = GameObject.FindGameObjectWithTag("ReferenceCross");
            if (ref_cross != null)
            {
                ref_cross.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0.5f);
                //ref_cross.GetComponent<RectTransform>().transform.localScale = new Vector3(2e-05f, 2e-05f, 1f);
            }
            
            // StartCoroutine(InitializeDB(trial, head1, head2, head3));            
        }
        else
        {
            // Load more GameObjects UI
            // head rotator initializations
            f_sphere1 = GameObject.FindGameObjectWithTag("Sphere1");
            f_sphere2 = GameObject.FindGameObjectWithTag("Sphere2");
            f_sphere3 = GameObject.FindGameObjectWithTag("Sphere3");

            r_sphere1 = GameObject.FindGameObjectWithTag("RotatorSphere1");
            r_sphere2 = GameObject.FindGameObjectWithTag("RotatorSphere2");
            r_sphere3 = GameObject.FindGameObjectWithTag("RotatorSphere3");

            plane1 = GameObject.FindGameObjectWithTag("Plane1");
            plane2 = GameObject.FindGameObjectWithTag("Plane2");
            plane3 = GameObject.FindGameObjectWithTag("Plane3");

            // initializing head and camera positions
            head1_3D.transform.position = new Vector3(924, head1_3D.transform.position.y, head1_3D.transform.position.z);
            head1_3D.transform.eulerAngles = new Vector3(0, 180, 0);
            f_sphere1.transform.position = new Vector3(924, f_sphere1.transform.position.y, f_sphere1.transform.position.z);
            camera1.transform.position = new Vector3(924, camera1.transform.position.y, camera1.transform.position.z);
            plane1.transform.position = new Vector3(924, plane1.transform.position.y, plane1.transform.position.z);

            head2_3D.transform.position = new Vector3(960, head2_3D.transform.position.y, head2_3D.transform.position.z);
            head2_3D.transform.eulerAngles = new Vector3(0, 180, 0);
            f_sphere2.transform.position = new Vector3(960, f_sphere2.transform.position.y, f_sphere2.transform.position.z);
            camera2.transform.position = new Vector3(960, camera2.transform.position.y, camera2.transform.position.z);
            plane2.transform.position = new Vector3(960, plane2.transform.position.y, plane2.transform.position.z);

            head3_3D.transform.position = new Vector3(996, head3_3D.transform.position.y, head3_3D.transform.position.z);
            head3_3D.transform.eulerAngles = new Vector3(0, 180, 0);
            f_sphere3.transform.position = new Vector3(996, f_sphere3.transform.position.y, f_sphere3.transform.position.z);
            camera3.transform.position = new Vector3(996, camera3.transform.position.y, camera3.transform.position.z);
            plane3.transform.position = new Vector3(996, plane3.transform.position.y, plane3.transform.position.z);

            // Game Object UIs
            text_feed = GameObject.FindGameObjectWithTag("Feed");
            text_feed.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0.0f);

            end_training = GameObject.FindGameObjectWithTag("EndTraining");
            end_training.GetComponent<Text>().enabled = false;

            end_block = GameObject.FindGameObjectWithTag("EndTestBlock");
            end_block.GetComponent<Text>().enabled = false;

            // reference cross
            ref_cross = GameObject.FindGameObjectWithTag("ReferenceCross");
            if (ref_cross != null)
            {
                ref_cross.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0.0f);
            }

            // textfeed.GetComponent<Text>().text = "Choose the odd one out";
            // this.stim_indicator = GameObject.FindGameObjectWithTag("StimIndicator");
            // if (stim_indicator != null)
            // {
            //    stim_indicator.GetComponent<Renderer>().enabled = false;
            //    stim_indicator.GetComponent<Renderer>().material.color = new Color(1f, 1f, 1f, 0f);
            // }

            if (t_end != 0)
            {
                StartCoroutine(SelectOddOneOut(trial));
            }
        }
    }

    IEnumerator InitializeDB(Trial trial, int head1, int head2, int head3)
    {
        if (exit_exp)
        {
            StopAllCoroutines();
        }
        ref_cross.GetComponent<Text>().text = "TripletsDB set, saving data...";

        yield return new WaitForSeconds(2f);
        exit_exp = true;
        EndIfCaughtExit(trial);
    }

    public IEnumerator ShowSlide(string slideName)
    {
        Text slideText = GameObject.Find(slideName).GetComponent<Text>();
        slideText.enabled = true;
        yield return new WaitForSeconds(0.2f); // prevent accidental press with other keys or previous trial
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
        slideText.enabled = false;
    }

    public IEnumerator AttentionInfo()
    {
        yield return StartCoroutine(ShowSlide("CatchTrialTraining"));
        Session.instance.CurrentTrial.End();
    }

    public IEnumerator BeginTraining()
    {
        yield return StartCoroutine(ShowSlide("BeginTraining"));
        Session.instance.BeginNextTrial();
    }

    public void CoBeginTraining()
    {
        StartCoroutine(BeginTraining());
    }

    IEnumerator SelectOddOneOut(Trial trial)
    {
        if (caught || exit_exp)
        {
            StopAllCoroutines();
            heads3D.SetActive(false);
            text_feed.SetActive(false);
        }
        // reference cross mark time
        ref_cross.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0.5f);
        yield return new WaitForSeconds(trial.settings.GetFloat("crossTrialTime"));
        ref_cross.GetComponent<Text>().color = defaultColor;

        /*
        // set initial wait time before training
        if (trial.number == 1)
        {
            ref_cross.GetComponent<TextMesh>().color = new Color(1f, 1f, 0, 0.8f);
            ref_cross.GetComponent<TextMesh>().text = "Loading your training session, Staring soon...";
            yield return new WaitForSeconds(2);
            ref_cross.GetComponent<TextMesh>().color = new Color(1f, 1f, 1f, 0f);
        }    

        // set wait time after training and before experiment
        if (trial.number == 5)
        {
            ref_cross.GetComponent<TextMesh>().color = new Color(1f, 1f, 0, 0.8f);
            // ref_cross.GetComponent<TextMesh>().transform.
            ref_cross.GetComponent<TextMesh>().text = "Loading the experiment, this may take a minute...";
            yield return new WaitForSeconds(2);
            ref_cross.GetComponent<TextMesh>().color = new Color(1f, 1f, 1f, 0f);
        }
        */

        // stim_indicator.GetComponent<Renderer>().enabled = true;
        // stim_indicator.GetComponent<Renderer>().material.color = new Color(0.78f, 0.78f, 0, 0.2f);

        // render head stimuli
        head1_3D.GetComponent<MeshRenderer>().enabled = true;
        head2_3D.GetComponent<MeshRenderer>().enabled = true;
        head3_3D.GetComponent<MeshRenderer>().enabled = true;

        if (trial.number <= GenerateExperiment.n_training_trials)
        {
            text_feed.GetComponent<Text>().enabled = true;
            text_feed.GetComponent<Text>().color = new Color(1f, 1f, 0f, 0.5f);
            text_feed.GetComponent<Text>().alignment = TextAnchor.UpperCenter;

            if (trial.number >= 3)
            {
                text_feed.GetComponent<Text>().text = "OBSERVE\nThis is an ATTENTION TEST";
                yield return new WaitForSeconds(1f);
            }
            else
            {
                text_feed.GetComponent<Text>().text = "OBSERVE";
            }
            yield return new WaitForSeconds(1f);
        }
        // buffer observation time
        yield return new WaitForSeconds(t_stim);

        // allow input response and change stim_indicator color to green
        // stim_indicator.GetComponent<Renderer>().material.color = new Color(0, 1f, 0, 0.2f);
        // textfeed.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0.0f);

        while ((trial.number <= GenerateExperiment.n_training_trials ? true : t_curr <= t_end))
        {
            if (trial.number <= GenerateExperiment.n_training_trials)
            {
                text_feed.GetComponent<Text>().enabled = true;
                text_feed.GetComponent<Text>().color = new Color(0f, 1f, 0f, 0.5f);
                text_feed.GetComponent<Text>().alignment = TextAnchor.UpperCenter;

                if (trial.number < 3)
                {
                    text_feed.GetComponent<Text>().text = "Use the ARROW KEYS to select the MOST DISSIMILAR face\n\n[LEFT] left —— [DOWN] middle —— [RIGHT] right";
                }
                else
                {
                    text_feed.GetComponent<Text>().text = "Two faces are identical\nChoose the DISSIMILAR face";
                }
            }
            allowResponse();


            if (clicked1)
            {
                head_selected = head1;
            }
            if (clicked2)
            {
                head_selected = head2;
            }
            if (clicked3)
            {
                head_selected = head3;
            }
            if (clicked1 || clicked2 || clicked3) // allow entering only if selected
            {
                if (trial.number <= GenerateExperiment.n_training_trials)
                {
                    // Debug.Log(catch_head + " c s " + head_selected);
                    text_feed.GetComponent<Text>().enabled = true;
                    text_feed.GetComponent<Text>().color = new Color(0f, 1f, 0f, 0.5f);
                    text_feed.GetComponent<Text>().alignment = TextAnchor.LowerCenter;
                    if (trial.number < 3)
                    {
                        text_feed.GetComponent<Text>().text = "You successfully made a choice\nPress SPACEBAR to continue to the next trial";//YOU CAN CHOOSE ANOTHER\n OR PRESS ENTER TO CONFIRM";
                    }
                    else
                    {
                        if (catch_trial && catch_head != head_selected)
                        {
                            text_feed.GetComponent<Text>().color = new Color(1f, 0f, 0f, 0.5f);
                            text_feed.GetComponent<Text>().text = "You failed the ATTENTION TEST!\nPay more attention next time\n Press SPACEBAR to continue to the next trial";//YOU CAN CHOOSE ANOTHER\n OR PRESS ENTER TO CONFIRM";
                        }
                        else
                        {
                            text_feed.GetComponent<Text>().text = "Well done!\nPress SPACEBAR to continue";//YOU CAN CHOOSE ANOTHER\n OR PRESS ENTER TO CONFIRM";
                        }
                    }
                    // response feedback light
                    //yield return new WaitForSeconds(0.5f); // 500 milliseconds
                    yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
                }

                correct_resp = 1;
                StartCoroutine(endTrial(trial));
                yield break;
                // To check if the results are correct before logging
                // Debug.Log("heads: " + head1.ToString() + ", " + head2.ToString() + ", " + head3.ToString());
                // Debug.Log("head_selected: " + head_selected.ToString() + ", t_resp: " + t_resp.ToString() + ", correct_resp: " + correct_resp.ToString());
                // Debug.Log("Current trial number: " + Session.instance.currentTrialNum.ToString());            
            }
            yield return null;
        }
        // Time out
        if (!nextTrial)
        {
            // stim_indicator.GetComponent<Renderer>().material.color = new Color(0.78f, 0, 0, 0.2f);
            correct_resp = 0;
            text_feed.GetComponent<Text>().enabled = true;
            text_feed.GetComponent<Text>().color = new Color(1f, 0f, 0f, 0.5f);
            text_feed.GetComponent<Text>().alignment = TextAnchor.UpperCenter;
            if (trial.number <= GenerateExperiment.n_training_trials)
            {
                text_feed.GetComponent<Text>().text = "TIME OUT, press SPACEBAR to continue to next trial";    
            }
            else
            {
                text_feed.GetComponent<Text>().text = "TIME OUT, continuing to next trial";
                yield return new WaitForSeconds(0.5f);
            }
            StartCoroutine(endTrial(trial));
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Time update
        t_curr = t_curr + Time.deltaTime;

        // head rotators active
        if (GenerateExperiment.stim_3D && !caught && !exit_exp)
        {
            if (t_curr <= t_end)
            {
                rotate3D();
                
            }
        }
    }

    // Function to create catch heads
    void catchTheTrial(Trial trial)
    {
        int rand_pos = Random.Range(1, 13);
        if (rand_pos <= 4)
        {
            head3_3D = GameObject.Instantiate(head1_3D, heads3D.transform);
            trial.settings.SetValue("head3", head1);
            catch_head = head2;
        }
        else if (rand_pos <= 8)
        {
            head2_3D = GameObject.Instantiate(head1_3D, heads3D.transform);
            trial.settings.SetValue("head2", head1);
            catch_head = head3;
        }
        else
        {
            head3_3D = GameObject.Instantiate(head2_3D, heads3D.transform);
            trial.settings.SetValue("head3", head2);
            catch_head = head1;
        }
    }

    // 3D stimulus activation to rotate around
    void rotate3D()
    {
        if (!nextTrial
           && f_sphere1 != null && f_sphere2 != null && f_sphere3 != null
           && r_sphere1 != null && r_sphere2 != null && r_sphere3 != null
           && head1_3D != null && head2_3D != null && head3_3D != null)
        {
            rotateSpheres(f_sphere1, r_sphere1);
            rotateSpheres(f_sphere2, r_sphere2);
            rotateSpheres(f_sphere3, r_sphere3);

            rotateHead(r_sphere1, head1_3D);
            rotateHead(r_sphere2, head2_3D);
            rotateHead(r_sphere3, head3_3D);
        }
    }

    IEnumerator UpdateBreakTimer(float t_curr0, string originalText)
    {
        Text breakText = end_block.GetComponent<Text>();
        while (t_curr - t_curr0 < maxt_break)
        {
            breakText.text = breakText.text.Replace("{currentBlock}", $"{Session.instance.currentBlockNum - 1}");
            breakText.text = breakText.text.Replace("{totalBlocks}", $"{Session.instance.blocks.Count - 1}");
            breakText.text = breakText.text.Replace("{maxBreak}", $"{maxt_break - Mathf.Floor(t_curr - t_curr0)}");

            yield return new WaitForSeconds(1);
            breakText.text = originalText;
        }
    }

    // Function to save and reset a list of variables at the end of each trial
    IEnumerator endTrial(Trial trial) //endTrial()
    {
        t_resp = t_curr;
        trial.result["triplet_id"] = trial.settings.GetInt("triplet_id");
        trial.result["triplet"] = trial.settings.GetString("triplet");
        trial.result["head_odd"] = head_selected;
        trial.result["correct"] = correct_resp;
        trial.result["response_time"] = t_resp;
        trial.result["caught"] = catch_trial && head_selected != catch_head;
        trial.result["catch_head"] = catch_head;

        if (one_time_logger >= 1)
        {
            Session.instance.SaveText("Screen Res - DPI: " + Screen.currentResolution.ToString() + " - " + Screen.dpi.ToString(),"metadata.txt");
            one_time_logger -= 1;
        }

        // response feedback light
        if (trial.number > GenerateExperiment.n_training_trials)
        {
            yield return new WaitForSeconds(0.5f);
        }
        // yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Return));

        // resetting planes
        clicked1 = false;
        clicked2 = false;
        clicked3 = false;

        plane1.GetComponent<Renderer>().material.color = defaultColor;
        plane2.GetComponent<Renderer>().material.color = defaultColor;
        plane3.GetComponent<Renderer>().material.color = defaultColor;

        // resetting heads
        head1_3D.GetComponent<MeshRenderer>().enabled = false;
        head2_3D.GetComponent<MeshRenderer>().enabled = false;
        head3_3D.GetComponent<MeshRenderer>().enabled = false;

        // count catches
        if ((trial.number > GenerateExperiment.n_training_trials) && catch_trial && catch_head != head_selected)
        {
            catch_count += 1;
        }

        catch_head = 0;

        if (trial.number <= GenerateExperiment.n_training_trials)
        {
            //yield return new WaitForSeconds(2f); // 2 second extra for feedback reading
            //yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
        }
        if (trial.number == GenerateExperiment.n_training_trials)
        {
            // stim_indicator.GetComponent<Renderer>().enabled = false;
            text_feed.GetComponent<Text>().enabled = false;
            yield return StartCoroutine(ShowSlide("BonusPayment"));
            end_training.GetComponent<Text>().enabled = true;
            yield return new WaitForSeconds(0.2f); // prevent accidental press with other keys or previous trial
            yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
            end_training.GetComponent<Text>().enabled = false;
        }
       
        // end of blocks text
        if (trial.number > GenerateExperiment.n_training_trials)
        {
            if (trial.numberInBlock == GenerateExperiment.n_trialsblock && trial.number != trial.session.LastTrial.number) // activate on last trial in each block (except very last trial)
            {
                catch_sequence = Enumerable.Range(6, GenerateExperiment.n_trialsblock - 5).OrderBy(n => n * n * (new System.Random()).Next());
                catch_random3 = catch_sequence.Distinct().Take(3).ToList();
                // stim_indicator.GetComponent<Renderer>().enabled = false;
                text_feed.GetComponent<Text>().enabled = false;
                end_block.GetComponent<Text>().enabled = true;
                t_curr0 = t_curr;

                string originalText = end_block.GetComponent<Text>().text;
                Coroutine breakTimer = StartCoroutine(UpdateBreakTimer(t_curr0, originalText));
                yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space) || t_curr - t_curr0 >= maxt_break); // || Input.GetKeyDown(KeyCode.Escape) 
                t_break = t_curr - t_curr0;
                trial.result["break_time"] = t_break;
                StopCoroutine(breakTimer);
                end_block.GetComponent<Text>().text = originalText;

                // if subject chooses to quit through ESC button - MAKE IT MORE EXPLICIT AND UNLOCK ADTER 2 hours or so
                // if (Input.GetKeyDown(KeyCode.Escape))
                // {
                //     exit_exp = true;
                // }

                end_block.GetComponent<Text>().enabled = false;
            }
        }
        nextTrial = true;
        catch_trial = false;

        // danger if 3 missed catches
        if (catch_count >= 3 || exit_exp)
        {
            text_feed.GetComponent<Text>().enabled = true;
            text_feed.GetComponent<Text>().color = new Color(1f, 0f, 0f, 0.5f);
            text_feed.GetComponent<Text>().alignment = TextAnchor.MiddleCenter;

            if (exit_exp)
            {
                text_feed.GetComponent<Text>().text = "You ABORTED the experiment\nThank you for participating!";
            }
            else
            {
                caught = true;
                text_feed.GetComponent<Text>().text = "You missed 3 ATTENTION TESTS\nExperiment ABORTED\nThank you for participating!";
            }

            Debug.Log("Caught 3 times: Setting all triplets to unseen...");
            foreach (string triplet in GenerateExperiment.currentBlockDB.Keys)
            {
                int triplet_id = (int)((Dictionary<string, object>)GenerateExperiment.currentBlockDB[triplet])["triplet_id"];
                Debug.Log($"Resetting {triplet}, id: {triplet_id}");
                GenerateExperiment.setTripletsDB(triplet_id, "000_s001_tripletsdb", "U", triplet);
            }

            yield return new WaitForSeconds(2f); // || Input.GetKeyDown(KeyCode.Escape));
            EndIfCaughtExit(trial);
        }
        else
        {
            if (trial.number == 2)
            {
                text_feed.GetComponent<Text>().enabled = false;
                StartCoroutine(AttentionInfo());
            }
            else
            {
                trial.End();
            }
        }
    }

    // sphere rotators to guide head motion
    void rotateSpheres(GameObject target, GameObject rotator)
    {
        // sphere rotation around fixed spheres
        rotator.transform.RotateAround(target.transform.position, rotateDir, rot_time * 360 * Time.deltaTime);
    }

    // head rotation activations
    void rotateHead(GameObject target, GameObject rotator)
    {
        // faces following the rotating sphere
        Vector3 dir = target.transform.position - rotator.transform.position;
        Quaternion lookRotation = Quaternion.LookRotation(dir);
        Vector3 rotation = Quaternion.Lerp(rotator.transform.rotation, lookRotation, Time.deltaTime * 1000).eulerAngles;
        rotator.transform.rotation = Quaternion.Euler(rotation.x, rotation.y, rotation.z);
        // rotator.transform.position
    }

    // response listener function for each trial
    public void allowResponse()
    {
        // response listener
        if (mouse)
        {
            if (Input.GetMouseButtonDown(0))
            {
                clickPlanes();

                // check the object clicked on
                // Ray ray = camera.ScreenPointToRay(Input.mousePosition);
                // RaycastHit hit; // Debug.DrawRay(ray.origin, ray.direction * 1000, Color.yellow);
                // if (Physics.Raycast(ray, out hit)){  hitColor(ray, hit); }

                // Debugging clicks
                // Debug.Log("Clicked: " + clicked1.ToString() + clicked2.ToString() + clicked3.ToString());
            }
        }
        else // keyboard listener
        {
            Event e = Event.current;
            if (e.isKey)
            {
                if (e.type == EventType.KeyDown)
                {
                    if (lFace.Contains(e.keyCode) && !mFace.Contains(e.keyCode) && !rFace.Contains(e.keyCode))
                    {
                        plane1.GetComponent<Renderer>().material.color = activeColor;
                        clicked1 = true;
                        clicked2 = false;
                        clicked3 = false;
                        Session.instance.CurrentTrial.result["keyPress"] = e.keyCode.ToString();
                    }
                    if (!lFace.Contains(e.keyCode) && mFace.Contains(e.keyCode) && !rFace.Contains(e.keyCode))
                    {
                        plane2.GetComponent<Renderer>().material.color = activeColor;
                        clicked1 = false;
                        clicked2 = true;
                        clicked3 = false;
                        Session.instance.CurrentTrial.result["keyPress"] = e.keyCode.ToString();
                    }
                    if (!lFace.Contains(e.keyCode) && !mFace.Contains(e.keyCode) && rFace.Contains(e.keyCode))
                    {
                        clicked1 = false;
                        clicked2 = false;
                        clicked3 = true;
                        plane3.GetComponent<Renderer>().material.color = activeColor;
                        Session.instance.CurrentTrial.result["keyPress"] = e.keyCode.ToString();
                    }
                }
            }
        }
    }

    // response and feedback function of each trial 
    void clickPlanes()// void hitColor(Ray ray, RaycastHit hit)
    {
        // feedback of the clicked choices by color change

        // if (hit.transform != null) {GameObject go = hit.transform.gameObject; // print(go.name);
        if (Input.mousePosition.x < Screen.width / 3) // go.name == "Plane1")
        {
            if (!clicked1)
            {
                plane1.GetComponent<Renderer>().material.color = activeColor;
                plane2.GetComponent<Renderer>().material.color = defaultColor;
                plane3.GetComponent<Renderer>().material.color = defaultColor;
                clicked1 = true;
                clicked2 = false;
                clicked3 = false;
            }
            else
            {
                plane1.GetComponent<Renderer>().material.color = defaultColor;
                clicked1 = false;
            }

        }
        if (Input.mousePosition.x >= Screen.width / 3 && Input.mousePosition.x < Screen.width * 2 / 3)  // go.name == "Plane2")
        {
            if (!clicked2)
            {
                plane2.GetComponent<Renderer>().material.color = activeColor;
                plane1.GetComponent<Renderer>().material.color = defaultColor;
                plane3.GetComponent<Renderer>().material.color = defaultColor;
                clicked2 = true;
                clicked1 = false;
                clicked3 = false;
            }
            else
            {
                plane2.GetComponent<Renderer>().material.color = defaultColor;
                clicked2 = false;
            }

        }
        if (Input.mousePosition.x >= Screen.width * 2 / 3) // go.name == "Plane3")
        {
            if (!clicked3)
            {
                plane3.GetComponent<Renderer>().material.color = activeColor;
                plane2.GetComponent<Renderer>().material.color = defaultColor;
                plane1.GetComponent<Renderer>().material.color = defaultColor;
                clicked3 = true;
                clicked1 = false;
                clicked2 = false;
            }
            else
            {
                plane3.GetComponent<Renderer>().material.color = defaultColor;
                clicked3 = false;
            }
        }
    }

    // Central Dictionary Update Function
    public void UpdateDictionaryAfterBlock(Trial trial)
    {
        if(trial.number > GenerateExperiment.n_training_trials)        
        {
            // Update Dictionary after a block
            if (trial.numberInBlock == GenerateExperiment.n_trialsblock)
            {
                bool mark_all_bad = false;
                Block c_block = Session.instance.CurrentBlock;
                // Mark the entire block bad if caught

                int k = 1;
                Debug.Log("Block Number: " + c_block.number.ToString());
                Debug.Log("Loop variables: Limit - " + (c_block.lastTrial.number - c_block.firstTrial.number + 1).ToString());
                while (k - 1 <= (c_block.lastTrial.number - c_block.firstTrial.number))
                {
                    Trial c_trial = c_block.GetRelativeTrial(k);
                    Debug.Log("Iterator: " + k.ToString());
                    Debug.Log("Heads: " + c_trial.settings.GetInt("head1").ToString() + ", " + c_trial.settings.GetInt("head2").ToString() + ", " + c_trial.settings.GetInt("head3").ToString());

                    int keyDB = c_trial.settings.GetInt("triplet_id");
                    string triplet = c_trial.settings.GetString("triplet");
                    k++;

                    if ((bool)c_trial.result["caught"])
                    {
                        Debug.Log("Trial number: " + c_trial.number.ToString() + " got caught");
                        mark_all_bad = true;
                        break;
                    }
                    else if ((int)c_trial.result["catch_head"] != 0)  // Add the original triplet that got converted to catch trial back to the unseen DB
                    {

                        ((Dictionary<string,object>)GenerateExperiment.currentBlockDB[triplet])["status"] = "U";
                        GenerateExperiment.setTripletsDB(keyDB, "000_s001_tripletsdb", "U", triplet);
                        Debug.Log($"Setting catch back to unseen. Triplet ({keyDB}): {triplet}, Status: U");
                    }
                    else
                    {
                        // Mark trial bad if no correct repsonse
                        if ((int)c_trial.result["correct"] == 0)
                        {
                            ((Dictionary<string,object>)GenerateExperiment.currentBlockDB[triplet])["status"] = "U";
                            GenerateExperiment.setTripletsDB(keyDB, "000_s001_tripletsdb", "U", triplet);
                            Debug.Log($"No correct response. Triplet ({keyDB}): {triplet}, Status: U");
                        }
                        else // Mark trial good otherwise
                        {
                            ((Dictionary<string,object>)GenerateExperiment.currentBlockDB[triplet])["status"] = "G";
                            GenerateExperiment.setTripletsDB(keyDB, "000_s001_tripletsdb", "G", triplet);
                            Debug.Log($"Triplet ({keyDB}): {triplet}, Status: G");
                        }
                    }
                }
                if (mark_all_bad)         // If caught, mark whole block bad
                {
                    Debug.Log("Marking all trials of this block bad");
                    for (int i = 1; i <= c_block.lastTrial.number - c_block.firstTrial.number + 1; i++)
                    {
                        Trial c_trial = c_block.GetRelativeTrial(i);
                        int keyDB = c_trial.settings.GetInt("triplet_id");
                        string triplet = c_trial.settings.GetString("triplet");
                        Debug.Log($"Bad block. Triplet ({keyDB}): {triplet}, Status: U");
                        GenerateExperiment.setTripletsDB(keyDB, "000_s001_tripletsdb", "U", triplet);
                        ((Dictionary<string,object>)GenerateExperiment.currentBlockDB[triplet])["status"] = "U";
                    }
                }
            }
        }
    }

    // Last trial
    public void EndIfLastTrial(Trial trial)
    {
        // Last trial of the session, configuring the end
        text_feed.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0f);
        // stim_indicator.GetComponent<Renderer>().enabled = false;

        if (trial == Session.instance.LastTrial)
        {
            if (!caught && !exit_exp) Session.instance.End();
            if (Input.GetKeyDown(KeyCode.Space))
            {
                // Press Space Button to quit
                Application.Quit();
            }
        }
    }

    public void EndIfCaughtExit(Trial trial)
    {
        text_feed.GetComponent<Text>().color = new Color(1f, 1f, 1f, 0f);
        if (caught || exit_exp)
        {            
            // Debug.Log("Ending Session");
            Session.instance.End();
        }
    }

    public void Redirect()
    {
        GameObject.Find("GoodBye").GetComponent<Text>().text += "\n\nPlease wait. You will be redirected after a few seconds...";
        string targetURL;
        // Completed
        if (!caught && !exit_exp) targetURL = "https://app.prolific.co/submissions/complete?cc=C1MT6VRN";
        // Failed attention checks
        else targetURL = "https://app.prolific.co/submissions/complete?cc=C1FX264D";
        StartCoroutine(RedirectURLTimer(targetURL));
    }

    public IEnumerator RedirectURLTimer(string targetURL)
    {
        yield return new WaitForSeconds(3f);
        RedirectURL(targetURL);
    }

    public void RedirectURL(string targetURL)
    {
        //string url = "https://www.cbs.mpg.de/departments/neurology/mind-body-emotion";
        //Application.ExternalEval($"window.open('{url}','_self')");
        //Application.OpenURL("https://www.cbs.mpg.de/departments/neurology/mind-body-emotion");
        OpenURL(targetURL);
    }

    // public void EndIfNotFullScreen(Trial trial)
}
